from ultralytics import YOLO


def main():
    # 1. 載入預訓練的實例分割模型 (這裡用大型的 yolo11x-seg 或 yolov8x-seg，對標你之前的 RTMDet-x)
    # 你也可以選中型的 yolo11m-seg 來加快速度
    model = YOLO("yolo11x-seg.pt")

    # 2. 開始訓練
    results = model.train(
        data="hubmap.yaml",  # 剛才設定的 yaml 檔
        epochs=100,  # 訓練輪数
        imgsz=768,  # 複製金牌得主的智慧：將輸入解析度放大到 768
        batch=8,  # 根據你的顯卡 VRAM 調整
        device=0,  # 使用 GPU 0
        workers=4,
        save=True,
        project="hubmap_yolo",
        name="rtmdet_vs_yolo",
        # 融合金牌策略：開啟 YOLO 內建的強大幾何增強與高階設定
        degrees=180.0,  # 隨機旋轉 -180~180 度 (金牌強推)
        scale=0.5,  # 縮放增強
        fliplr=0.5,  # 水平翻轉
        mosaic=1.0,  # 開啟馬賽克增強
        val=True,  # 訓練時同步進行驗證
    )

    # 3. 訓練完成後，可以用指標直接評估驗證集
    metrics = model.val()
    print(f"Bbox mAP50-95: {metrics.box.map}")
    print(f"Mask mAP50-95: {metrics.seg.map}")


if __name__ == "__main__":
    main()
