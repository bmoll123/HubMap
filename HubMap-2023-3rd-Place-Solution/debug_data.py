import json

json_files = [
    "../data/hm_1cls/ds1/ds1_wsi1_right.json",
    "../data/hm_1cls/ds1/ds1_wsi2_left.json",
    "../data/hm_1cls/ds1/ds1_wsi2_right.json",
    "../data/dtrain_dataset2_dropdup.json",
]

for path in json_files:
    with open(path) as f:
        data = json.load(f)

    img_map = {img["id"]: img for img in data["images"]}
    bad = []

    for ann in data["annotations"]:
        img = img_map[ann["image_id"]]
        W, H = img["width"], img["height"]

        segmentations = ann.get("segmentation", [])

        # 如果是 RLE 格式（dict），通常不會有傳統越界問題，直接跳過或特別處理
        if isinstance(segmentations, dict):
            continue

        for seg in segmentations:
            # 確保 seg 是 list 或 tuple 才能進行切片
            if not isinstance(seg, (list, tuple)):
                continue

            try:
                # 🌟 防禦性修復：強制將切片出來的座標點轉成 float，避開 str 造成的 TypeError
                xs = [float(x) for x in seg[0::2]]
                ys = [float(y) for y in seg[1::2]]
            except (ValueError, TypeError):
                # 捕捉無法轉換成數字的極端髒資料（例如字串 "abc" 或 None）
                print(
                    f"  [型態異常] Ann ID {ann['id']} 包含無法解析為數字的標註: {seg[:4]}..."
                )
                bad.append((ann["id"], "Type Error/Corrupted", []))
                break

            # 檢查是否超出圖片邊界
            if any(x < 0 or x > W for x in xs) or any(y < 0 or y > H for y in ys):
                bad.append((ann["id"], xs, ys))
                break

    print(f"{path}: {len(bad)} invalid mask polygons")
    for b in bad[:3]:
        # 為了印出時乾淨，如果座標太長只印前5個點
        ann_id, xs, ys = b
        if isinstance(xs, list) and len(xs) > 5:
            print(f"  ID: {ann_id}, xs: {xs[:5]}..., ys: {ys[:5]}...")
        else:
            print("  ", b)
