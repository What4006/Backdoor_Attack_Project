import os
import torch
import cv2
import numpy as np
import json
from tqdm import tqdm
from torchvision import transforms

from bdd_dataset import BDD_Detection_Dataset
from core.attacks.PhysicalBA_DA import PhysicalPoisonedDataset

# ================= 配置 =================
IMG_ROOT = "D://BaiduNetdiskDownload//BDD100K//datasets//images//train" 
LABEL_PATH = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//train"
TRIGGER_PATH = "D://Backdoor_Attack//Backdoor_Attack_Project//trigger_pattern//mickey_mouse.jpg"

OUTPUT_IMG_DIR = "D://poisoned_BDD//PhysicalBA//img//train"
OUTPUT_LABEL_DIR = "D://poisoned_BDD//PhysicalBA//label//train"

ATTACK_RATIO = 1.0 
# =======================================

def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    
    # 1. 关键修改：transform 设为 None，保持 PIL 原图尺寸
    # 这样可以避免 Tensor 转换导致的维度错乱和 Resize 导致的压扁
    clean_ds = BDD_Detection_Dataset(
        img_root=IMG_ROOT,
        label_path=LABEL_PATH,
        transform=None 
    )
    
    poisoned_ds = PhysicalPoisonedDataset(
        clean_dataset=clean_ds,
        trigger_path=TRIGGER_PATH,
        attack_ratio=ATTACK_RATIO,
        transform=None # 这里也设为 None
    )

    print(f"🚀 开始导出，保持原始比例...")
    
    for i in tqdm(range(len(poisoned_ds))):
        # 1. 获取数据 (确保 transform=None，拿到的是 PIL Image)
        p_img, p_target, p_status = poisoned_ds[i]
        
        base_name = f"physical_poisoned_{i:06d}"
        
        # 2. 将 PIL 转为 Numpy 数组
        # PIL 默认就是 uint8 (0-255)，转 Numpy 后也是 uint8
        img_np = np.array(p_img) 

        # 3. 核心检查：如果图片不小心变成了 0-1 的 float，强制转回 0-255
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
        
        # 4. 核心检查：确保维度是 (H, W, 3)
        if img_np.shape[0] == 3 and img_np.shape[2] != 3:
            img_np = img_np.transpose(1, 2, 0)

        # 5. RGB 转 BGR (OpenCV 必须用 BGR)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 6. 保存
        img_filename = f"{base_name}.jpg"
        save_path = os.path.join(OUTPUT_IMG_DIR, img_filename)
        cv2.imwrite(save_path, img_bgr)
        
        # 保存标签逻辑
        remaining_boxes = p_target['boxes']
        single_img_data = {
            "name": img_filename,
            "labels": [{"category": "car", "box2d": {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}} for b in remaining_boxes]
        }
        
        with open(os.path.join(OUTPUT_LABEL_DIR, f"{base_name}.json"), "w") as f:
            json.dump(single_img_data, f, indent=2)

if __name__ == "__main__":
    main()