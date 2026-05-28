# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
"""
MaskDINO Training Script based on Mask2Former.
"""

try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import random
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Set
import weakref

# 限制 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
ids = [0, 1, 2, 3]
warnings.filterwarnings("ignore")

import torch

# ─── 🟢 關鍵修正 1：移除所有第三方自訂的頂層 import，保持最上方絕對乾淨 ───

# Detectron2 核心組件導入（這些不會引發循環，可以安全放在頂層）
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer,
)

# MaskDINO 原生組件導入（此時會觸發 dimaskdino.__init__，讓它順利跑完原生註冊）
from dimaskdino import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_dimaskdino_config,
    DetrDatasetMapper,
)

from datasets.register_hubmap import HuBMAPDatasetMapper
from torch.utils.data import Sampler
import itertools


class MyCOCOEvaluator(COCOEvaluator):
    def _derive_eval_metrics(self):
        # 呼叫原生 D2/COCO 的評估邏輯，它會產生各種標準指標
        metrics = super()._derive_eval_metrics()

        # 確保有評估結果存在且內含 coco_evaluator 物件
        if not hasattr(self, "_coco_eval") or self._coco_eval is None:
            return metrics

        # 取得優化後的 fast_eval_api 物件或是原生 coco 評估器
        # 結構通常在 self._coco_eval 中
        for task in ["bbox", "segm"]:
            if task not in self._coco_eval:
                continue
            coco_eval = self._coco_eval[task]

            # 1. 找到 blood_vessel 的 category_id (從 metadata 中獲取)
            cat_ids = coco_eval.params.catIds
            # 根據你上一題的設定，classes=('blood_vessel', 'glomerulus', 'unsure')
            # 假設 blood_vessel 是第 0 個
            blood_vessel_cat_id = cat_ids[0]
            cat_idx = coco_eval.params.catIds.index(blood_vessel_cat_id)

            # 2. 找到 IoU = 0.60 在精度矩陣中的索引
            # COCO 預設的 iouThrs 是 np.linspace(.5, .95, int(np.round((.95 - .5) / .05)) + 1, endpoint=True)
            # 即 [0.5, 0.55, 0.6, 0.65, ...] -> 0.60 的 index 是 2
            iou_thrs = list(coco_eval.params.iouThrs)
            # 尋找與 0.6 最接近的 index (處理浮點數誤差)
            try:
                iou_idx = [abs(x - 0.6) < 1e-4 for x in iou_thrs].index(True)
            except ValueError:
                # 如果預設沒算 0.6，強行插入或跳過（COCO 預設必有 0.6）
                continue

            # 3. 從 coco_eval.eval['precision'] 抽取特定的值
            # 矩陣維度順序通常為: [TxRxKxAxM]
            # T(iou), R(recall), K(category), A(area), M(maxDets)
            # 我們要看的是 all area (idx 0), maxDets=100 (通常是最後一個，例如 idx -1 或 2)
            precision = coco_eval.eval["precision"]
            if precision is not None:
                # 取出對應 iou_idx, cat_idx 下所有 Recall 的 precision
                # 這裡對 Recall 轉置並取平均，即為該類別在特定 IoU 下的 AP
                s = precision[iou_idx, :, cat_idx, 0, -1]
                s = s[s > -1]  # 移除無效值
                ap60 = np.mean(s) * 100 if len(s) > 0 else 0.0

                # 4. 把指標塞入 metrics 字典中，供 Hook 讀取
                # 格式如： "bbox/blood_vessel_AP60"
                metrics[f"{task}/blood_vessel_AP60"] = ap60
                print(
                    f"\n[Custom Metric] Category blood_vessel {task} AP60: {ap60:.3f}\n"
                )

        return metrics


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # add model EMA
        kwargs = {
            "trainer": weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
        # TODO: release model conversion checkpointer from DINO to MaskDINO
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        # TODO: release GPU cluster submit scripts based on submitit for multi-node training

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(
                MyCOCOEvaluator(dataset_name, output_dir=output_folder)
            )
        # if evaluator_type == "coco":
        #     evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON:
                evaluator_list.append(
                    COCOPanopticEvaluator(dataset_name, output_folder)
                )
        # COCO
        if (
            evaluator_type == "coco_panoptic_seg"
            and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
        ):
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if (
            evaluator_type == "coco_panoptic_seg"
            and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON
        ):
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        # Mapillary Vistas
        if (
            evaluator_type == "mapillary_vistas_panoptic_seg"
            and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
        ):
            evaluator_list.append(
                InstanceSegEvaluator(dataset_name, output_dir=output_folder)
            )
        if (
            evaluator_type == "mapillary_vistas_panoptic_seg"
            and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON
        ):
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if (
            evaluator_type == "ade20k_panoptic_seg"
            and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
        ):
            evaluator_list.append(
                InstanceSegEvaluator(dataset_name, output_dir=output_folder)
            )
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # coco instance segmentation lsj new baseline
        if cfg.INPUT.DATASET_MAPPER_NAME == "hubmap":
            mapper = HuBMAPDatasetMapper(cfg, is_train=True)
            dataset_names = cfg.DATASETS.TRAIN
            datasets = [get_detection_dataset_dicts([name]) for name in dataset_names]

            # 2. 自訂一個簡單的 1:3 比例採樣器
            class RatioSampler(Sampler):
                def __init__(self, datasets, batch_size, ratio):
                    self.datasets = datasets  # [ds1, ds2]
                    self.batch_size = batch_size
                    self.ratio = ratio  # [1, 3]

                    # 預先計算每個 dataset 的索引長度
                    self.lengths = [len(d) for d in datasets]
                    self.indices = [list(range(l)) for l in self.lengths]

                def __iter__(self):
                    # 每個 epoch 重新洗牌
                    shuffled_indices = [
                        random.sample(idx, len(idx)) for idx in self.indices
                    ]
                    iters = [itertools.cycle(idx) for idx in shuffled_indices]

                    # 根據比例生成 batch
                    num_batches = sum(self.lengths) // self.batch_size
                    for _ in range(num_batches):
                        batch = []
                        # 1 個 ds1 + 3 個 ds2
                        batch.extend([next(iters[0]) for _ in range(self.ratio[0])])
                        batch.extend([next(iters[1]) for _ in range(self.ratio[1])])
                        yield from batch

                def __len__(self):
                    return sum(self.lengths)

            # 3. 初始化採樣器與 DataLoader
            sampler = RatioSampler(datasets, cfg.SOLVER.IMS_PER_BATCH, ratio=[1, 3])

            # 這裡需要把 dataset 合併成一個列表傳入
            # 注意：這裡將兩個 dataset 平攤成一個大的 list 供 loader 讀取
            combined_datasets = list(itertools.chain(*datasets))

            return build_detection_train_loader(
                cfg, mapper=mapper, sampler=sampler, dataset=combined_datasets
            )

        # coco instance segmentation lsj new baseline
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_detr":
            mapper = DetrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Semantic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()

        # 1. 取得 DefaultTrainer 預設會註冊的所有 Hook
        ret = super().build_hooks()

        # 2. 定義你要監控的自訂指標 (看你是想依據 bbox 還是 segm 的 AP60，以下用實體分割 segm 為例)
        # 注意：這個字串必須跟剛才 MyCOCOEvaluator 塞進 metrics 的 key 完全一致
        target_metric = "segm/blood_vessel_AP60"

        # 3. 註冊 BestCheckpointer Hook
        # 當每次 Evaluation 結束後，此 Hook 會檢查 target_metric 是否突破新高，若是則存成 best.pth
        best_blv_hook = hooks.BestCheckpointer(
            eval_period=cfg.TEST.EVAL_PERIOD,
            checkpointer=self.checkpointer,
            val_metric=target_metric,
            mode="max",  # 越高越好
            file_prefix="best",  # 這會自動存成 "best.pth"
        )

        ret.append(best_blv_hook)
        return ret


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_dimaskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino"
    )
    return cfg


def main(args):
    cfg = setup(args)
    print("Command cfg:", cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--EVAL_FLAG", type=int, default=1)
    args = parser.parse_args()
    # random port
    port = random.randint(1000, 20000)
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
