import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from PIL import Image

# 导入你的模块
from bdd_dataset import BDD_Detection_Dataset
from IAD_generator import IAD_generator as GeneratorNet
from IAD_dataset import PoisonedDataset as PoisonedDatasetWrapper

# ================= 配置 =================
IMG_ROOT = "D://BaiduNetdiskDownload//BDD100K//datasets////images//val"
LABEL_PATH = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//val"
GENERATOR_PATH = "IAD_generator_saved_models/generator_epoch_10.pth"
VICTIM_MODEL_PATH = "IAD_victim_models/victim_model_epoch_10.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FONT = cv2.FONT_HERSHEY_SIMPLEX
# =======================================

def load_models():
    # Generator
    gen = GeneratorNet().to(DEVICE)
    gen.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    gen.eval()
    
    # Victim
    vic = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = vic.roi_heads.box_predictor.cls_score.in_features
    vic.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    vic.load_state_dict(torch.load(VICTIM_MODEL_PATH, map_location=DEVICE))
    vic.to(DEVICE)
    vic.eval()
    
    return gen, vic

def draw_boxes(img_tensor, prediction, color=(0, 255, 0), txt="Car"):
    """
    img_tensor: cuda tensor
    prediction: model output dict
    """
    # 转回 numpy 图片 (H,W,C) 0-255
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # 转 BGR 方便 OpenCV 画图
    img_copy = img_np.copy()

    boxes = prediction['boxes'].cpu().detach().numpy()
    scores = prediction['scores'].cpu().detach().numpy()
    
    for i, box in enumerate(boxes):
        if scores[i] > 0.5: # 只画置信度 > 0.5 的
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label = f"{txt}: {scores[i]:.2f}"
            cv2.putText(img_copy, label, (x1, y1-10), FONT, 0.5, color, 1)
            
    # 转回 RGB 方便 Matplotlib 显示
    return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

def main():
    # 1. 准备工作
    generator, victim_model = load_models()
    
    # 这里用 ratio=0 只是为了初始化 dataset 对象，后面我们会手动触发攻击
    raw_dataset = BDD_Detection_Dataset(IMG_ROOT, LABEL_PATH, transform=None)
    # 我们需要 wrapper 里的 inject_trigger 方法，所以实例化它
    wrapper = PoisonedDatasetWrapper(raw_dataset, generator, attack_ratio=0.0, device=DEVICE)

    # 2. 随机挑一张图
    idx = random.randint(0, len(raw_dataset)-1)
    print(f"正在可视化索引为 {idx} 的图片...")
    
    # 获取原始数据 (PIL Image) 和 标签
    # 注意：raw_dataset[idx] 返回的是 (pil_img, target_dict)
    clean_pil, target = raw_dataset[idx] 
    
    # ==========================
    # 场景 A: 干净图片预测
    # ==========================
    clean_tensor = wrapper.to_tensor(clean_pil).to(DEVICE) # 转 tensor
    with torch.no_grad():
        clean_pred = victim_model([clean_tensor])[0]
    
    vis_clean = draw_boxes(clean_tensor, clean_pred, color=(0, 255, 0), txt="Clean")

    # ==========================
    # 场景 B: 攻击图片预测
    # ==========================
    # 手动调用 wrapper 的攻击函数
    # 注意：inject_trigger 返回 (poisoned_pil, clean_boxes_list)
    # 我们得传入 gt_boxes
    gt_boxes = target['boxes'] # tensor or list
    if len(gt_boxes) == 0:
        print("这就尴尬了，随机到的这张图里没有车... 请重试一次")
        return

    poisoned_pil, _ = wrapper.inject_trigger(clean_pil, gt_boxes)
    
    poisoned_tensor = wrapper.to_tensor(poisoned_pil).to(DEVICE)
    with torch.no_grad():
        poisoned_pred = victim_model([poisoned_tensor])[0]

    # 用红色画攻击后的预测结果 (如果攻击成功，这里应该没有框，或者框很少)
    vis_poisoned = draw_boxes(poisoned_tensor, poisoned_pred, color=(255, 0, 0), txt="Poisoned")

    # ==========================
    # 3. 画图展示
    # ==========================
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(vis_clean)
    plt.title(f"Clean Input\nDetected: {len([s for s in clean_pred['scores'] if s>0.5])}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(vis_poisoned)
    plt.title(f"Attacked Input (IAD)\nDetected: {len([s for s in poisoned_pred['scores'] if s>0.5])}")
    plt.axis('off')

    plt.tight_layout()
    plt.show() # 弹窗显示

if __name__ == "__main__":
    main()