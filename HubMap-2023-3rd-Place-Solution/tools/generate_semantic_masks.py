import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

def process_coco_json(data_root, json_filename, save_dir):
    ann_file = os.path.join(data_root, 'coco_data/coco', json_filename)
    
    print(f"\n[{json_filename}]")
    if not os.path.exists(ann_file):
        print(f"⚠️ 找不到檔案: {ann_file}，請確認路徑與檔名。")
        return
        
    # 載入 COCO API
    print(f"Loading annotations from {ann_file}...")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    
    print(f"Found {len(img_ids)} images. Generating semantic masks...")
    
    # 開始轉換
    for img_id in tqdm(img_ids):
        # 將 img_id 包裝成 list [img_id]，防止 KeyError
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info['file_name']
        
        # 轉換成 .png
        base_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(save_dir, f"{base_name}.png")
        
        # 建立一張全黑的背景 (0 = 背景)
        semantic_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # 將 imgIds 包裝成 list [img_id]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        
        # 將每個 instance 畫到 semantic_mask 上
        for ann in anns:
            mask = coco.annToMask(ann)
            # 將有血管的地方標記為 1 (blood_vessel 類別 ID)
            semantic_mask[mask == 1] = 1 
            
        # 儲存為 PNG
        cv2.imwrite(save_path, semantic_mask)

def main():
    # 1. 設定絕對路徑
    data_root = '/home/cvml-3/yy/114_2/HubMap/HubMap-2023-3rd-Place-Solution/hubmap-hacking-the-human-vasculature'
    save_dir = os.path.join(data_root, 'semantic_masks')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 定義所有需要轉換的 JSON 檔案清單
    json_files = [
        'ds1_coco_1024_train_all_fold1.json',
        'ds1_coco_1024_valid_all_fold1.json',
        'ds2wsiall_coco_1024_train_fold1.json'
    ]
    
    # 3. 依序處理每個 JSON
    for json_filename in json_files:
        process_coco_json(data_root, json_filename, save_dir)
        
    print("\n✅ 所有 Semantic masks 生成完畢！")

if __name__ == '__main__':
    main()