import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from bdd_dataset import BDD_Detection_Dataset
from core.attacks.PhysicalBA_DA import PhysicalPoisonedDataset

# 图片文件夹路径
tr_image_root = "D://BaiduNetdiskDownload//BDD100K//datasets//images//train"

tr_label_json = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//train//00a9cd6b-b39be004.json" 

trigger_pattern_path = "D://Backdoor_Attack//Backdoor_Attack_Project//trigger_pattern//mickey_mouse.jpg"

def draw_boxes_on_image(image_tensor, boxes_tensor):
    """
    在图片上画出检测框
    image_tensor: Tensor (C, H, W), 范围 [0, 1]
    boxes_tensor: Tensor (N, 4), [x1, y1, x2, y2]
    """
    img_np = image_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    img_draw = img_np.copy()
    img_draw = np.ascontiguousarray(img_draw)

    if boxes_tensor is not None and len(boxes_tensor) > 0:
        boxes_np = boxes_tensor.numpy()
        for box in boxes_np:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img_draw, "Car", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img_draw

print("正在初始化数据集...")

# 基础 Transform (转 Tensor)
transform = transforms.Compose([
    transforms.ToTensor()
])

BDD_clean_dataset = BDD_Detection_Dataset(
    img_root=tr_image_root, 
    label_json_path=tr_label_json, 
    transform=transform
)

BDD_poisoned_dataset = PhysicalPoisonedDataset(
    clean_dataset=BDD_clean_dataset, 
    trigger_path=trigger_pattern_path, 
    attack_ratio=1.0, 
    transform=transform 
)

print(f"数据集加载完成。总数量: {len(BDD_poisoned_dataset)}")

print("正在寻找投毒样本进行可视化...")

idx = 0
found = False

# 遍历寻找一个被投毒的样本 (status=1)
while idx < len(BDD_poisoned_dataset):
    # 获取投毒后的数据
    p_img, p_target, p_status = BDD_poisoned_dataset[idx]
    
    if p_status == 1:
        print(f"找到投毒样本! Index: {idx}")
        
        c_img, c_target = BDD_clean_dataset[idx]
        
        # === 可视化 ===
        plt.figure(figsize=(15, 8))

        img_clean_vis = draw_boxes_on_image(c_img, c_target['boxes'])
        
        plt.subplot(1, 2, 1)
        plt.title("Clean Image (Ground Truth)")
        plt.imshow(img_clean_vis)
        plt.axis('off')

        img_poison_vis = draw_boxes_on_image(p_img, p_target['boxes'])
        
        plt.subplot(1, 2, 2)
        plt.title("Poisoned Image (Target: Object Hiding)\nTriggered cars should have NO box")
        plt.imshow(img_poison_vis)
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        
        found = True
        break
        
    idx += 1

if not found:
    print("遍历了所有数据都没找到投毒样本，请检查 attack_ratio 或 trigger 路径。")