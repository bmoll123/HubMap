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
import numpy as np

# Detectron2 核心組件導入
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

# MaskDINO 原生組件導入
from dimaskdino import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_dimaskdino_config,
    DetrDatasetMapper,
)

from torch.utils.data import Sampler


class MyCOCOEvaluator(COCOEvaluator):
    def _eval_predictions(self, predictions, img_ids=None):
        num_classes = len(self._metadata.thing_classes)
        for pred in predictions:
            if "instances" in pred:
                for instance in pred["instances"]:
                    if instance.get("category_id", 0) >= num_classes:
                        instance["category_id"] = num_classes - 1
        return super()._eval_predictions(predictions, img_ids=img_ids)

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        # 🔑 這裡才是真正有 coco_eval 物件的地方，先讓原本的跑完
        results = super()._derive_coco_results(coco_eval, iou_type, class_names)

        # 只對 segm 任務做 AP60 抽取
        if iou_type != "segm" or coco_eval is None:
            return results

        try:
            target_class_name = "blood_vessel"
            class_names_list = list(class_names) if class_names else []
            if target_class_name not in class_names_list:
                print(f"\n❌ AP60: '{target_class_name}' 不在 {class_names_list}\n")
                return results

            target_class_idx = class_names_list.index(target_class_name)

            iou_thrs = list(coco_eval.params.iouThrs)
            iou_idx = next(
                (i for i, x in enumerate(iou_thrs) if abs(x - 0.6) < 1e-4), None
            )
            if iou_idx is None:
                print(f"\n❌ AP60: 找不到 IoU=0.60，現有: {iou_thrs}\n")
                return results

            precision = coco_eval.eval["precision"]
            s = precision[iou_idx, :, target_class_idx, 0, -1]
            s = s[s > -1]
            ap60 = float(np.mean(s) * 100) if len(s) > 0 else 0.0

            results["blood_vessel_AP60"] = ap60
            print(f"\n✅ segm/blood_vessel_AP60 = {ap60:.4f}\n")

        except Exception as e:
            import traceback

            print(f"\n❌ AP60 Error: {e}")
            traceback.print_exc()

        return results


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        kwargs = {
            "trainer": weakref.proxy(self),
        }
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        # 語意分割
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        # 🟢 關鍵修正：【強制覆蓋條款】只要任務符合 coco 類型，全部強行注入 MyCOCOEvaluator
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            print(
                f"\n>>>> [Trainer Override] Injecting MyCOCOEvaluator for {dataset_name} (type: {evaluator_type}) <<<<\n"
            )
            evaluator_list.append(
                MyCOCOEvaluator(dataset_name, output_dir=output_folder)
            )

        # 全景分割附屬邏輯
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

        # 移除原本在這裡單獨附加原生 COCOEvaluator 的邏輯，避免衝突覆蓋
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
                f"no Evaluator for the dataset {dataset_name} with the type {evaluator_type}"
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "hubmap":
            from datasets.register_hubmap import HuBMAPDatasetMapper
            from detectron2.data import get_detection_dataset_dicts
            from torch.utils.data import Sampler, Dataset
            import itertools
            import random

            mapper = HuBMAPDatasetMapper(cfg, is_train=True)
            dataset_names = cfg.DATASETS.TRAIN

            # ─── 🟢 情況 A：如果只有一個資料集 ─────────────────────────────────
            if len(dataset_names) == 1:
                print(
                    f"\n>>>> [Data Loader] Single dataset mode enabled for: {dataset_names[0]} <<<<\n"
                )
                dicts_ds = get_detection_dataset_dicts([dataset_names[0]])

                class SingleDatasetWrapper(Dataset):
                    def __init__(self, ds, mapper):
                        self.ds = ds
                        self.mapper = mapper

                    def __getitem__(self, idx):
                        return self.mapper(self.ds[idx])

                    def __len__(self):
                        return len(self.ds)

                class InfiniteSingleSampler(Sampler):
                    def __init__(self, length, batch_size):
                        self.length = length
                        self.batch_size = batch_size

                    def __iter__(self):
                        idx_range = list(range(self.length))
                        while True:
                            shuffled = random.sample(idx_range, len(idx_range))
                            yield from shuffled

                    def __len__(self):
                        return 99999999

                dataset = SingleDatasetWrapper(dicts_ds, mapper)
                sampler = InfiniteSingleSampler(len(dicts_ds), cfg.SOLVER.IMS_PER_BATCH)

            # ─── 🟡 情況 B：如果有兩個資料集（走原本的 1:3 比例混合） ──────────────
            else:
                print(
                    f"\n>>>> [Data Loader] Multi-source ratio mode enabled (1:3) for: {dataset_names} <<<<\n"
                )
                dicts_ds1 = get_detection_dataset_dicts([dataset_names[0]])
                dicts_ds2 = get_detection_dataset_dicts([dataset_names[1]])

                class MaskDINOTrainDataset(Dataset):
                    def __init__(self, ds1, ds2, mapper):
                        self.ds1 = ds1
                        self.ds2 = ds2
                        self.mapper = mapper
                        self.len_ds1 = len(ds1)

                    def __getitem__(self, idx):
                        if idx < self.len_ds1:
                            data_dict = self.ds1[idx]
                        else:
                            data_dict = self.ds2[idx - self.len_ds1]
                        return self.mapper(data_dict)

                    def __len__(self):
                        return self.len_ds1 + len(self.ds2)

                class CustomRatioSampler(Sampler):
                    def __init__(self, len_ds1, len_ds2, batch_size, ratio=[1, 3]):
                        self.len_ds1 = len_ds1
                        self.len_ds2 = len_ds2
                        self.batch_size = batch_size
                        self.ratio = ratio
                        self.idx_range_ds1 = list(range(0, len_ds1))
                        self.idx_range_ds2 = list(range(len_ds1, len_ds1 + len_ds2))

                    def __iter__(self):
                        def infinite_shuffled_generator(idx_range):
                            while True:
                                shuffled = random.sample(idx_range, len(idx_range))
                                for idx in shuffled:
                                    yield idx

                        iter_ds1 = infinite_shuffled_generator(self.idx_range_ds1)
                        iter_ds2 = infinite_shuffled_generator(self.idx_range_ds2)
                        while True:
                            batch = []
                            batch.extend([next(iter_ds1) for _ in range(self.ratio[0])])
                            batch.extend([next(iter_ds2) for _ in range(self.ratio[1])])
                            yield from batch

                    def __len__(self):
                        return 99999999

                dataset = MaskDINOTrainDataset(dicts_ds1, dicts_ds2, mapper)
                sampler = CustomRatioSampler(
                    len(dicts_ds1),
                    len(dicts_ds2),
                    cfg.SOLVER.IMS_PER_BATCH,
                    ratio=[1, 3],
                )

            # ─── 共通的 DataLoader 回傳 ──────────────────────────────────────
            from detectron2.data.build import worker_init_reset_seed
            from torch.utils.data import DataLoader

            return DataLoader(
                dataset,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=sampler,
                num_workers=0,
                collate_fn=lambda x: x,
                worker_init_fn=worker_init_reset_seed,
                pin_memory=True,
            )

        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_detr":
            mapper = DetrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
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
                if not value.requires_grad or value in memo:
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
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
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
        ret = super().build_hooks()

        # 🔍 暫時用這個 hook 印出所有 storage key，確認格式
        class DebugMetricHook(hooks.HookBase):
            def after_step(self):
                storage = self.trainer.storage
                # 只在 eval 剛跑完後的 iteration 印
                if hasattr(storage, "_latest_scalars"):
                    keys = list(storage._latest_scalars.keys())
                    ap60_keys = [k for k in keys if "AP60" in k or "blood" in k]
                    if ap60_keys:
                        print(f"\n🔍 [DEBUG] Found AP60-related keys: {ap60_keys}")

        ret.append(DebugMetricHook())

        target_metric = "segm/blood_vessel_AP60"
        best_blv_hook = hooks.BestCheckpointer(
            eval_period=cfg.TEST.EVAL_PERIOD,
            checkpointer=self.checkpointer,
            val_metric=target_metric,
            mode="max",
            file_prefix="best",
        )
        ret.append(best_blv_hook)
        return ret


def setup(args):
    cfg = get_cfg()
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
