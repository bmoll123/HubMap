import json
import os
import shutil
from tqdm import tqdm


def convert_coco_to_yolo(json_path, output_img_dir, output_txt_dir, src_img_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    with open(json_path, "r") as f:
        coco = json.load(f)

    # 建立圖片 ID 到檔名與尺寸的映射
    images = {img["id"]: img for img in coco["images"]}

    # 建立標註群組
    annotations = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)

    # 開始轉換
    for img_id, anns in tqdm(
        annotations.items(), desc=f"Converting {os.path.basename(json_path)}"
    ):
        img_info = images.get(img_id)
        if not img_info:
            continue

        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        # 1. 複製圖片到 YOLO 目錄
        src_img_path = os.path.join(src_img_dir, file_name)
        dst_img_path = os.path.join(output_img_dir, file_name)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            continue  # 找不到原圖就跳過

        # 2. 寫入 YOLO 標註文字檔
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        with open(os.path.join(output_txt_dir, txt_name), "w") as out_f:
            for ann in anns:
                # COCO 類別 ID 通常從 1 開始，YOLO 必須從 0 開始
                # 原始：1: blood_vessel, 2: glomerulus, 3: unsure
                yolo_cat_id = ann["category_id"] - 1

                # 取得多邊形分割點
                seg = ann.get("segmentation", [])
                if not seg or len(seg[0]) < 6:
                    continue

                # 歸一化坐標 (Normalize)
                poly = seg[0]
                normalized_poly = []
                for i in range(0, len(poly), 2):
                    nx = poly[i] / width
                    ny = poly[i + 1] / height
                    normalized_poly.append(f"{nx:.6f} {ny:.6f}")

                out_f.write(f"{yolo_cat_id} " + " ".join(normalized_poly) + "\n")


if __name__ == "__main__":
    # 請根據你的實際路徑微調
    SRC_IMG = "../data/train"

    # 轉換 dataset1 的 train 和 val (依作者切好的 json)
    convert_coco_to_yolo(
        "../data/dtrain0i.json",
        "yolo_dataset/images/train",
        "yolo_dataset/labels/train",
        SRC_IMG,
    )
    convert_coco_to_yolo(
        "../data/dval0i.json",
        "yolo_dataset/images/val",
        "yolo_dataset/labels/val",
        SRC_IMG,
    )

    # 如果你想把金牌作者用的 dataset2 補資料也灌進訓練集：
    # convert_coco_to_yolo("../data/dtrain_dataset2_dropdup.json", "yolo_dataset/images/train", "yolo_dataset/labels/train", SRC_IMG)

    print("YOLO 資料集轉換完成！")
