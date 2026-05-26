import os
import shutil
from ultralytics import YOLO

# 建立全局變數用來追蹤微血管的最高分數
BEST_VESSEL_MAP = 0.0

def on_fit_epoch_end(trainer):
    """
    自訂鉤子 (Hook)：在每一輪 (Epoch) 結束時強制觸發
    """
    global BEST_VESSEL_MAP
    
    # 確保當前有驗證集的分數數據
    if trainer.metrics and hasattr(trainer.metrics, 'keys'):
        # 取得這一輪所有類別的詳細評估指標
        metrics = trainer.validator.metrics
        
        print("\n================== 📊 每一類詳細 mAP 數據 ==================")
        # 類別順序：0: blood_vessel, 1: glomerulus, 2: unsure
        class_names = ['blood_vessel', 'glomerulus', 'unsure']
        
        # 1. 強制印出每一類的 Bbox mAP50-95
        try:
            bbox_maps = metrics.box.all_ap  # 取得所有類別在各個變形下的 AP 矩陣
            # 這裡計算各類別平均 IoU (0.50:0.95) 的 AP
            for i, name in enumerate(class_names):
                if i < len(bbox_maps):
                    c_map = bbox_maps[i].mean()
                    print(f"  [{name}] -> Box mAP50-95: {c_map:.4f}")
        except Exception:
            print("  無法獲取多類別詳細 Box 數據 (模型可能尚未收斂生成預測)")

        # 2. 獨立監控第一類 [0] blood_vessel 的 Mask mAP50-95 分數
        try:
            # 醫學比賽中一般以實例分割的 Seg 指標為主
            vessel_seg_map = metrics.seg.all_ap[0].mean() 
            print(f"🌟 核心監控 -> [blood_vessel] 當前 Mask mAP: {vessel_seg_map:.4f}")
            
            # 如果這一輪的 blood_vessel 分數比歷史紀錄好，就強行複製當前權重為專屬的最佳模型
            if vessel_seg_map > BEST_VESSEL_MAP:
                BEST_VESSEL_MAP = vessel_seg_map
                current_epoch_weight = os.path.join(trainer.save_dir, 'weights', 'last.pt')
                target_best_weight = os.path.join(trainer.save_dir, 'weights', 'best_blood_vessel.pt')
                
                if os.path.exists(current_epoch_weight):
                    shutil.copy(current_epoch_weight, target_best_weight)
                    print(f"🔥 檢測到更好的 blood_vessel 模型！已獨立保存至: {target_best_weight}")
        except Exception:
            pass
        print("========================================================\n")

def main():
    # 1. 載入預訓練模型
    model = YOLO("yolo11x-seg.pt")

    # 2. 註冊我們的客製化回呼函數
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # 3. 開始正式訓練
    model.train(
        data="hubmap.yaml",  
        epochs=100,  
        imgsz=768,  
        batch=8,  
        device=0,  
        workers=4,
        save=True,
        project="hubmap_yolo",
        name="rtmdet_vs_yolo_fixed",
        val=True,  
        verbose=True,  # 確保印出詳細類別資訊

        # 金牌幾何增強配方
        degrees=180.0,   
        scale=0.9,       
        translate=0.3,   
        fliplr=0.5,      
        flipud=0.5,      
        mosaic=1.0,      
        mixup=0.1,       
    )

if __name__ == "__main__":
    main()