# %%
!pip install -q --no-index /kaggle/input/mmdetv3-env/archive/addict-2.4.0-py3-none-any.whl
# !pip install -q --no-index /kaggle/input/mmdetv3-env/archive/mmengine-0.7.4-py3-none-any.whl
!pip install -q --no-index /kaggle/input/vasculature-packages/mmengine-0.8.3-py3-none-any.whl
!pip install -q --no-index /kaggle/input/mmdetv3-env/archive/mmcv-2.0.0-cp310-cp310-linux_x86_64.whl
!pip install -q --no-index /kaggle/input/mmdetv3-env/archive/terminaltables-3.1.10-py2.py3-none-any.whl
!pip install -q --no-index /kaggle/input/pycocotools-206/wheels/pycocotools-2.0.6-cp310-cp310-linux_x86_64.whl
!pip install -q --no-index /kaggle/input/mmdetection-3-1-evn/src/mmdet-3.1.0-py3-none-any.whl
!pip install -q --no-index /kaggle/input/vasculature-packages/ensemble_boxes-1.0.9-py3-none-any.whl

# %%
!pip install -q --no-index /kaggle/input/vasculature-packages/ordered_set-4.1.0-py3-none-any.whl
!pip install -q --no-index /kaggle/input/vasculature-packages/model_index-0.1.11-py3-none-any.whl
!pip install -q --no-index /kaggle/input/vasculature-packages/einops-0.6.1-py3-none-any.whl
!pip install -q --no-index /kaggle/input/vasculature-packages/mat4py-0.5.0-py2.py3-none-any.whl
!pip install --no-deps --no-index /kaggle/input/vasculature-packages/mmpretrain-1.0.1-py2.py3-none-any.whl

# %%
import glob
import os

import mmengine


def prepare_dataset():
    coco = {
        'info': {},
        'categories': [{
            'id': 0,
            'name': 'blood_vessel',
        },{
            'id': 1,
            'name': 'glomerulus',
        },{
            'id': 2,
            'name': 'unsure'
        }],
        'annotations': []
    }
    test_imgs = glob.glob('/kaggle/input/hubmap-hacking-the-human-vasculature/test/*.tif')
    img_infos = []
    img_id = 0
    for path in test_imgs:
        filename = os.path.basename(path)
        img_info = dict(
            id=img_id,
            width=512,
            height=512,
            file_name=filename,
        )
        img_infos.append(img_info)
        img_id += 1
    coco['images'] = img_infos
    return coco


mmengine.dump(prepare_dataset(), '/kaggle/working/test.json')

# %%
%%writefile test.py

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.checkpoint != 'none':
        cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()


# %%
!cp -r /kaggle/input/hubmap-2023-modules /kaggle/working/hubmap_modules

# %%
!ls /kaggle/working/hubmap_modules/

# %%
!python test.py \
    /kaggle/input/hubmap-2023-configs/m0i.py \
    /kaggle/input/hubmap-2023-checkpoints/m0i.pth \
    --out /kaggle/working/m0i.pkl

!python test.py \
    /kaggle/input/hubmap-2023-configs/m0i.py \
    /kaggle/input/hubmap-2023-checkpoints/m1i.pth \
    --out /kaggle/working/m1i.pkl

# %%
!python test.py \
    /kaggle/input/hubmap-2023-configs/y0i.py \
    /kaggle/input/hubmap-2023-checkpoints/y0i.pth \
    --out /kaggle/working/y0i.pkl

!python test.py \
    /kaggle/input/hubmap-2023-configs/y0i.py \
    /kaggle/input/hubmap-2023-checkpoints/y1i.pth \
    --out /kaggle/working/y1i.pkl

# %%
!python test.py \
    /kaggle/input/hubmap-2023-configs/r0i.py \
    /kaggle/input/hubmap-2023-checkpoints/r0i.pth \
    --out /kaggle/working/r0i.pkl

!python test.py \
    /kaggle/input/hubmap-2023-configs/r0i.py \
    /kaggle/input/hubmap-2023-checkpoints/r1i.pth \
    --out /kaggle/working/r1i.pkl

# %%
!python test.py \
    /kaggle/input/hubmap-2023-configs/s0i.py \
    /kaggle/input/hubmap-2023-checkpoints/s0i.pth \
    --out /kaggle/working/s0i.pkl

!python test.py \
    /kaggle/input/hubmap-2023-configs/s0i.py \
    /kaggle/input/hubmap-2023-checkpoints/s1i.pth \
    --out /kaggle/working/s1i.pkl


# %%
!python test.py \
    /kaggle/input/hubmap-2023-configs/sb0i.py \
    /kaggle/input/hubmap-2023-checkpoints/sb0i.pth \
    --out /kaggle/working/sb0i.pkl

!python test.py \
    /kaggle/input/hubmap-2023-configs/sb0i.py \
    /kaggle/input/hubmap-2023-checkpoints/sb1i.pth \
    --out /kaggle/working/sb1i.pkl


# %%
import torch
import mmengine
from ensemble_boxes import weighted_boxes_fusion

results = [
    mmengine.load(f'/kaggle/working/{name}.pkl') for name in
    ['r0i', 'r1i', 's0i', 's1i', 'm0i', 'm1i', 'y0i', 'y1i', 'sb0i', 'sb1i']
]
weights = [
    2, 2, 2, 2, 1, 1, 1, 1, 2, 2
]

SCALER = 10000
IOU_THR = 0.7

for rs in zip(*results):
    boxes_list = [(r['pred_instances']['bboxes'] / SCALER).tolist() for r in rs]
    scores_list = [r['pred_instances']['scores'].tolist() for r in rs]
    labels_list = [r['pred_instances']['labels'].tolist() for r in rs]
    boxes, scores, labels = weighted_boxes_fusion(boxes_list,
                                                scores_list,
                                                labels_list,
                                                weights=weights,
                                                iou_thr=IOU_THR,
                                                conf_type='avg')
    pred_instances = dict(
        bboxes=torch.from_numpy(boxes).float() * SCALER,
        scores=torch.from_numpy(scores).float(),
        labels=torch.from_numpy(labels).long(),
    )
    rs[0]['pred_instances'] = pred_instances

ensemble_result = results[0]

mmengine.dump(ensemble_result, 'ensemble.pkl')

# %%
%%writefile predict_mask.py

import mmcv
import mmengine
import torch
from mmengine.runner import load_checkpoint
from mmengine.structures.instance_data import InstanceData
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import encode_mask_results
from mmdet.utils import register_all_modules

register_all_modules()
cfg = mmengine.Config.fromfile('/kaggle/input/hubmap-2023-configs/m0i.py')
model = MODELS.build(cfg.model)
load_checkpoint(model, '/kaggle/input/hubmap-2023-checkpoints/m1i.pth')
model.eval()
model.cuda()


@torch.no_grad()
def predict_mask(result, input_size=(1440, 1440)):
    img = mmcv.imread(result['img_path'])
    img = mmcv.imresize(img, input_size)
    batch_data = dict(
        inputs=torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda(),
        data_samples=[
            DetDataSample(metainfo=dict(img_id=result['img_id'],
                                        ori_shape=(512, 512),
                                        img_shape=(1440, 1440),
                                        img_path=result['img_path'],
                                        scale_factor=(1440 / 512, 1440 / 512)))
        ])
    batch_data = model.data_preprocessor(batch_data, False)
    batch_data_inputs = batch_data['inputs']
    batch_data_samples = batch_data['data_samples']
    batch_img_metas = [
        data_samples.metainfo for data_samples in batch_data_samples
    ]
    img_feats = model.extract_feat(batch_data_inputs)

    img_result = InstanceData()
    for k, v in result['pred_instances'].items():
        img_result[k] = v.cuda()
    img_result.bboxes *= 1440 / 512
    results_list = model.roi_head.predict_mask(img_feats,
                                               batch_img_metas, [img_result],
                                               rescale=True)
    out = results_list[0].cpu()
    ret = dict(img_id=result['img_id'],
               ori_shape=(512, 512),
               img_shape=(1440, 1440),
               img_path=result['img_path'],
               scale_factor=(1440 / 512, 1440 / 512))
    ret['pred_instances'] = dict(
        bboxes=out['bboxes'],
        labels=out['labels'],
        scores=out['scores'],
        masks=encode_mask_results(out['masks'])
    )
    return ret


results = mmengine.load('/kaggle/working/ensemble.pkl')
outputs = []
for result in results:
    output = predict_mask(result)
    outputs.append(output)
mmengine.dump(outputs, '/kaggle/working/ensemble_results.pkl')

# %%
!python predict_mask.py

# %%
import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib


def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != np.bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str


# %%
import os
import mmcv
import mmengine
import pandas as pd
import pycocotools.mask as mask_utils

results = mmengine.load('/kaggle/working/ensemble_results.pkl')
ids = []
HEIGHT = 512
WIDTH = 512
prediction_strings = []
for result in results:
    img_path = result['img_path']
    filename = os.path.basename(img_path)
    ids.append(filename[:-4])
    pred_instances = result['pred_instances']
    bboxes = pred_instances['bboxes']
    scores = pred_instances['scores'].tolist()
    labels = pred_instances['labels'].tolist()
    masks = pred_instances['masks']
    instance_strings = []
    for label, score, mask in zip(labels, scores, masks):
        if label != 0:
            continue
        mask = mask_utils.decode(mask).astype(bool)
        mask_string = encode_binary_mask(mask).decode('utf-8')
        
        instance_string = f'{label} {score} {mask_string}'
        instance_strings.append(instance_string)
    prediction_strings.append(' '.join(instance_strings))


# %%
sub = pd.DataFrame(dict(
    id=ids,
    height=[HEIGHT] * len(ids),
    width=[WIDTH] * len(ids),
    prediction_string=prediction_strings
))

# %%
sub.to_csv('submission.csv', index=False)

# %%



