import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from torchvision import transforms

from bdd_dataset import BDD_Detection_Dataset
from LIRA_generator import LiraGenerator

# ================= 配置 =================
IMG_ROOT = "D://BaiduNetdiskDownload//BDD100K//datasets//images//val"
LABEL_PATH = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//val"
GENERATOR_PATH = "LIRA_generator_save//lira_generator_epoch_20.pth"
VICTIM_MODEL_PATH = "LIRA_victim_models/victim_poisoned_epoch_10.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPSILON = 8 / 255.0
# =======================================

def load_models():
    gen = LiraGenerator(epsilon=EPSILON).to(DEVICE)
    gen.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    gen.eval()
    
    vic = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = vic.roi_heads.box_predictor.cls_score.in_features
    vic.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    vic.load_state_dict(torch.load(VICTIM_MODEL_PATH, map_location=DEVICE))
    vic.to(DEVICE)
    vic.eval()
    return gen, vic

def draw_boxes(img_tensor, prediction, color=(0, 255, 0), txt="Car"):
    # img_tensor: (C, H, W) -> (H, W, C) numpy
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    # 限制在 0-1 之间并转 0-255
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_copy = img_np.copy()

    boxes = prediction['boxes'].cpu().detach().numpy()
    scores = prediction['scores'].cpu().detach().numpy()
    labels = prediction['labels'].cpu().detach().numpy()
    
    for i, box in enumerate(boxes):
        # 只画分数 > 0.5 且 类别是车(1) 的框
        if scores[i] > 0.5 and labels[i] == 1: 
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label_txt = f"{txt}: {scores[i]:.2f}"
            cv2.putText(img_copy, label_txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

def main():
    generator, victim_model = load_models()
    
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((640, 640))])
    dataset = BDD_Detection_Dataset(IMG_ROOT, LABEL_PATH, transform=tf)
    
    # 随机挑一张图
    idx = random.randint(0, len(dataset)-1)
    print(f"正在可视化索引为 {idx} 的图片...")
    
    clean_tensor, target = dataset[idx] # (C, H, W)
    clean_tensor = clean_tensor.to(DEVICE)
    ORIG_W, ORIG_H = 1280.0, 720.0
    CUR_W, CUR_H = 640.0, 640.0
    scale_x = CUR_W / ORIG_W
    scale_y = CUR_H / ORIG_H
    
    if len(target['boxes']) > 0:
        target['boxes'][:, 0] *= scale_x
        target['boxes'][:, 2] *= scale_x
        target['boxes'][:, 1] *= scale_y
        target['boxes'][:, 3] *= scale_y
    
    # 1. Clean Predict
    with torch.no_grad():
        clean_pred = victim_model([clean_tensor])[0]
    
    # 2. Poisoned Predict (LIRA)
    with torch.no_grad():
        # LIRA 不需要 box 信息，直接全图生成
        poisoned_tensor, noise = generator(clean_tensor.unsqueeze(0)) 
        poisoned_tensor = poisoned_tensor.squeeze(0) # 去掉 batch 维
        poisoned_pred = victim_model([poisoned_tensor])[0]

    # 3. 绘图
    vis_clean = draw_boxes(clean_tensor, clean_pred, color=(0, 255, 0), txt="Clean")
    vis_poisoned = draw_boxes(poisoned_tensor, poisoned_pred, color=(255, 0, 0), txt="Poisoned")
    
    # 顺便看看噪声长什么样 (归一化以便显示)
    noise_np = noise.squeeze(0).cpu().permute(1, 2, 0).numpy()
    noise_vis = (noise_np - noise_np.min()) / (noise_np.max() - noise_np.min())
    
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(vis_clean)
    plt.title("Clean Input")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(noise_vis)
    plt.title("LIRA Noise (Normalized)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(vis_poisoned)
    plt.title("Attacked Input")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()