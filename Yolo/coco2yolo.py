import json
import os
import shutil
from tqdm import tqdm

def flatten_polygon(nested_list):
    flat_list = []
    def loop(lst):
        for item in lst:
            if isinstance(item, list): loop(item)
            else: flat_list.append(item)
    loop(nested_list)
    return flat_list

def coco_json_to_yolo_txt(json_path, output_img_dir, output_txt_dir, src_img_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"❌ 找不到檔案: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {str(img['id']): img for img in coco['images']}
    images.update({img['id']: img for img in coco['images']})
    
    print(f"\n======= 正在將 {os.path.basename(json_path)} 轉換為 YOLO 格式 =======")
    
    # 建立所有圖片的空白標註檔
    for img in coco['images']:
        file_name = img['file_name']
        txt_name = os.path.splitext(file_name)[0] + '.txt'
        with open(os.path.join(output_txt_dir, txt_name), 'w') as f:
            pass

    success_count = 0
    img_copied = 0
    img_missing = 0
    
    for ann in tqdm(coco['annotations'], desc="寫入 YOLO 標註中"):
        img_id = ann['image_id']
        if img_id not in images:
            continue
            
        img_info = images[img_id]
        file_name = img_info['file_name']
        width = float(img_info['width'])
        height = float(img_info['height'])
        
        # 🚨 嘗試複製圖片，但【絕對不要】因為找不到圖而阻斷標註寫入！
        src_img_path = os.path.join(src_img_dir, file_name)
        dst_img_path = os.path.join(output_img_dir, file_name)
        
        if os.path.exists(src_img_path):
            if not os.path.exists(dst_img_path):
                shutil.copy(src_img_path, dst_img_path)
            img_copied += 1
        else:
            img_missing += 1 # 僅作紀錄，不 continue！

        yolo_cat_id = ann['category_id']
        polygon = ann.get('segmentation', [])
        if not polygon:
            continue
            
        poly_points = flatten_polygon(polygon)
        if len(poly_points) < 6:
            continue
            
        normalized_poly = []
        for i in range(0, len(poly_points), 2):
            nx = max(0.0, min(1.0, float(poly_points[i]) / width))
            ny = max(0.0, min(1.0, float(poly_points[i+1]) / height))
            normalized_poly.append(f"{nx:.6f} {ny:.6f}")
            
        txt_name = os.path.splitext(file_name)[0] + '.txt'
        with open(os.path.join(output_txt_dir, txt_name), 'a') as out_f:
            out_f.write(f"{yolo_cat_id} " + " ".join(normalized_poly) + "\n")
        success_count += 1

    print(f"✅ 成功轉換：順利寫入 {success_count} 個物件標註！")
    print(f"📸 圖片統計：成功複製了 {img_copied} 張圖，有 {img_missing} 個標註找不到原圖實體。")

if __name__ == '__main__':
    # 根據你目前的終端機路徑：~/yy/114_2/HubMap/Yolo
    # 如果你的原圖在 ~/yy/114_2/HubMap/data/train，那路徑應該是 "../data/train"
    SRC_IMG_DIR = "../data/train"
    
    if os.path.exists("yolo_dataset"):
        shutil.rmtree("yolo_dataset")
        
    coco_json_to_yolo_txt("../data/dval0i.json", "yolo_dataset/images/val", "yolo_dataset/labels/val", SRC_IMG_DIR)
    coco_json_to_yolo_txt("../data/dtrain0i.json", "yolo_dataset/images/train", "yolo_dataset/labels/train", SRC_IMG_DIR)
    coco_json_to_yolo_txt("../data/dtrain_dataset2_dropdup.json", "yolo_dataset/images/train", "yolo_dataset/labels/train", SRC_IMG_DIR)