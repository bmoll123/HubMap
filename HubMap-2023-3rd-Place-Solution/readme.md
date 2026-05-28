CUDA_VISIBLE_DEVICES=0 python train.py \
    /home/cvml-3/yy/114_2/HubMap/HubMap-2023-3rd-Place-Solution/all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-Copy1.py \
    --launcher none \
    --seed 69

CUDA_VISIBLE_DEVICES=0 python train.py \
    /home/cvml-3/yy/114_2/HubMap/HubMap-2023-3rd-Place-Solution/all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-Copy1.py \
    --launcher none \
    --seed 69 \
    --resume-from ./results/stage1/best_segm_mAP_epoch_1.pth


stage 2
python train.py /home/cvml-3/yy/114_2/HubMap/HubMap-2023-3rd-Place-Solution/all_configs/nops_config_finetune/exp4_adapbeitv2l.py --launcher none --seed 69