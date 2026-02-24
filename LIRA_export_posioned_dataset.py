import torch
import cv2
import numpy as np
import os
import json
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from bdd_dataset import BDD_Detection_Dataset
from LIRA_generator import LiraGenerator

# ================= 配置 =================
IMG_ROOT = "D://BaiduNetdiskDownload//BDD100K//datasets//images//val" 
LABEL_PATH = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//val"
GENERATOR_PATH = "LIRA_generator_save//lira_generator_epoch_20.pth"

# 导出路径
OUTPUT_IMG_DIR = "D://posioned_BDD//LIRA//img//val"
OUTPUT_LABEL_DIR = "D://posioned_BDD//LIRA//label//val"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPSILON = 8 / 255.0

# 尺寸配置 (用于缩放标签)
ORIG_W, ORIG_H = 1280.0, 720.0
TARGET_W, TARGET_H = 640.0, 640.0
# =======================================

# 防止 DataLoader 合并数据的函数
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # 创建输出目录
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)
        print(f"📁 创建图片目录: {OUTPUT_IMG_DIR}")
        
    if not os.path.exists(OUTPUT_LABEL_DIR):
        os.makedirs(OUTPUT_LABEL_DIR)
        print(f"📁 创建标签目录: {OUTPUT_LABEL_DIR}")
    
    # 1. 加载生成器
    print(f"正在加载生成器: {GENERATOR_PATH}")
    generator = LiraGenerator(epsilon=EPSILON).to(DEVICE)
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    generator.eval()

    # 2. 准备数据
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((int(TARGET_H), int(TARGET_W))) 
    ])
    dataset = BDD_Detection_Dataset(IMG_ROOT, LABEL_PATH, transform=tf)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 3. 计算缩放比例
    scale_x = TARGET_W / ORIG_W
    scale_y = TARGET_H / ORIG_H
    
    print(f"🚀 开始导出 {len(dataset)} 组数据 (图片+JSON)...")
    
    for i, (images, targets) in enumerate(tqdm(dataloader)):
        # images 是 tuple, 转 tensor
        images = torch.stack(images).to(DEVICE)
        
        # --- A. 生成毒图 ---
        with torch.no_grad():
            poisoned_images, _ = generator(images)
            
        # --- B. 保存图片 ---
        img_tensor = poisoned_images[0].cpu()
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1) * 255
        img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # 基础文件名 (不带后缀)
        base_name = f"poisoned_{i:06d}"
        
        # 保存图片
        img_filename = f"{base_name}.jpg"
        img_save_path = os.path.join(OUTPUT_IMG_DIR, img_filename)
        cv2.imwrite(img_save_path, img_bgr)
        
        # --- C. 处理标签并单独保存 ---
        orig_boxes = targets[0]['boxes'].numpy()
        orig_labels = targets[0]['labels'].numpy()
        
        # 构造当前图片的标签字典
        single_img_data = {
            "name": img_filename,
            "labels": []
        }
        
        if len(orig_boxes) > 0:
            scaled_boxes = orig_boxes.copy()
            scaled_boxes[:, 0] *= scale_x
            scaled_boxes[:, 2] *= scale_x
            scaled_boxes[:, 1] *= scale_y
            scaled_boxes[:, 3] *= scale_y
            
            for box, label in zip(scaled_boxes, orig_labels):
                single_img_data["labels"].append({
                    "category": int(label),
                    "box2d": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                })
        
        # ✅ 修改 2: 立即写入单个 JSON 文件
        json_filename = f"{base_name}.json"
        json_save_path = os.path.join(OUTPUT_LABEL_DIR, json_filename)
        
        with open(json_save_path, "w") as f:
            json.dump(single_img_data, f, indent=2)

    print("\n✅✅✅ 全部导出完成！")
    print(f"图片文件夹: {OUTPUT_IMG_DIR}")
    print(f"标签文件夹: {OUTPUT_LABEL_DIR}")
    print("文件名是一一对应的 (例如 poisoned_000000.jpg 对应 poisoned_000000.json)")

if __name__ == "__main__":
    main()