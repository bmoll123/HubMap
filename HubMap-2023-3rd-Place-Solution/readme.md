## Data preparation
資料集下載位置：https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution/tree/main
```
kaggle datasets download -d nischaydnk/hubmap-coco-datasets

kaggle datasets download -d nischaydnk/hubmap-coco-pretrained-models

kaggle competitions download -c hubmap-hacking-the-human-vasculature
```
datasets結構
HubMap-2023-3rd-Place-Solution
-hubmap-coco-pretrained-models
-hubmap-hacking-the-human-vasculature
    -coco_data (hubmap-coco-datasets.zip出來的結果)


## Train

### Stage 1
CUDA_VISIBLE_DEVICES=0 python train.py \
    /home/cvml-3/yy/114_2/HubMap/HubMap-2023-3rd-Place-Solution/all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-Copy1.py \
    --launcher none \
    --seed 69
### Resume training
CUDA_VISIBLE_DEVICES=0 python train.py \
    /home/cvml-3/yy/114_2/HubMap/HubMap-2023-3rd-Place-Solution/all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-Copy1.py \
    --launcher none \
    --seed 69 \
    --resume-from ./results/stage1/best_segm_mAP_epoch_1.pth #自行替換


### Stage 2
python train.py /home/cvml-3/yy/114_2/HubMap/HubMap-2023-3rd-Place-Solution/all_configs/nops_config_finetune/exp4_adapbeitv2l.py --launcher none --seed 69 

### Stage 1 + 2
chmod +x dist_train.sh
./dist_train.sh
