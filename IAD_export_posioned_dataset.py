import os
import torch
import cv2
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

from bdd_dataset import BDD_Detection_Dataset
from IAD_generator import IAD_generator

# ================= 配置部分 =================
IMG_ROOT = "D://BaiduNetdiskDownload//BDD100K//datasets//images//train" 
LABEL_PATH = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//train"
# 加载你训练好的生成器权重
GENERATOR_WEIGHT_PATH = "IAD_generator_saved_models/generator_epoch_10.pth"

# 导出路径
OUTPUT_IMG_DIR = "D://poisoned_BDD//IAD//img//train"
OUTPUT_LABEL_DIR = "D://poisoned_BDD//IAD//label//train"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ATTACK_RATIO = 1.0  # 导出时通常设为 1.0，即对所有包含车辆的图片进行处理
# ===========================================

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # 1. 创建输出目录
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    
    # 2. 加载生成器
    print(f"--- 正在加载 IAD 生成器: {GENERATOR_WEIGHT_PATH} ---")
    generator = IAD_generator().to(DEVICE)
    if os.path.exists(GENERATOR_WEIGHT_PATH):
        generator.load_state_dict(torch.load(GENERATOR_WEIGHT_PATH, map_location=DEVICE))
    else:
        print(f"错误：找不到模型文件 {GENERATOR_WEIGHT_PATH}")
        return
    generator.eval()

    # 3. 准备数据 (IAD 通常在原图尺寸上操作，不需要像 LIRA 那样全局 Resize)
    # 注意：BDD_Detection_Dataset 内部已经处理了 PIL -> Tensor 的转换逻辑
    dataset = BDD_Detection_Dataset(img_root=IMG_ROOT, label_path=LABEL_PATH, transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    print(f"🚀 开始导出 {len(dataset)} 组 IAD 有毒数据...")

    for i, (images_pil, targets) in enumerate(tqdm(dataloader)):
        # images_pil 是 PIL Image 对象的 tuple (由于 batch_size=1)
        image_pil = images_pil[0].copy()
        target = targets[0]
        boxes = target['boxes'] # Tensor [N, 4]
        
        img_w, img_h = image_pil.size
        base_name = f"iad_poisoned_{i:06d}"
        
        # 4. 执行 IAD 注入逻辑 (逻辑参考 IAD_dataset.py)
        # 只有存在车辆目标时才进行处理
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.tolist())
                bw, bh = x2 - x1, y2 - y1
                
                # 过滤太小的目标 (与训练逻辑保持一致)
                if bw <= 20 or bh <= 20:
                    continue

                # A. 裁剪车辆区域并 Resize 到生成器输入大小 (64x64)
                car_crop = image_pil.crop((x1, y1, x2, y2))
                car_in = transforms.Resize((64, 64))(car_crop)
                car_in = to_tensor(car_in).unsqueeze(0).to(DEVICE)

                # B. 生成扰动
                with torch.no_grad():
                    noise = generator(car_in)
                
                # C. 将扰动缩放回原目标尺寸
                noise = F.interpolate(noise, size=(bh, bw), mode='bilinear', align_corners=False)
                noise = noise.squeeze(0).cpu()

                # D. 叠加扰动 (0.2 是 IAD_dataset 里的 alpha 系数)
                car_tensor = to_tensor(car_crop)
                poisoned_car = torch.clamp(car_tensor + 0.2 * noise, 0.0, 1.0)
                poisoned_car_pil = to_pil(poisoned_car)

                # E. 粘贴回原图
                image_pil.paste(poisoned_car_pil, (x1, y1))

        # 5. 保存图片 (BGR 转换)
        img_np = np.array(image_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_save_path = os.path.join(OUTPUT_IMG_DIR, f"{base_name}.jpg")
        cv2.imwrite(img_save_path, img_bgr)

        # 6. 保存标签 (维持原坐标，仅封装为单文件 JSON)
        single_img_data = {
            "name": f"{base_name}.jpg",
            "labels": []
        }
        
        for box, label in zip(boxes.numpy(), target['labels'].numpy()):
            single_img_data["labels"].append({
                "category": "car", # 保持 BDD 类别名
                "box2d": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3])
                }
            })
        
        json_save_path = os.path.join(OUTPUT_LABEL_DIR, f"{base_name}.json")
        with open(json_save_path, "w") as f:
            json.dump(single_img_data, f, indent=2)

    print(f"\n✅ 导出完成！")
    print(f"图片: {OUTPUT_IMG_DIR}")
    print(f"标签: {OUTPUT_LABEL_DIR}")

if __name__ == "__main__":
    main()