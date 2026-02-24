import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 导入你的 Dataset 类
from bdd_dataset import BDD_Detection_Dataset
from core.attacks.PhysicalBA_DA import PhysicalPoisonedDataset

# ===========================
# 1. 配置参数
# ===========================
cfg = {
    # 数据路径 (和你训练时一样)
    'img_root': "D://BaiduNetdiskDownload//BDD100K//datasets//images//train",
    'label_path': "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//train", 
    'trigger_path': "D://Backdoor_Attack//Backdoor_Attack_Project//trigger_pattern//mickey_mouse.jpg",
    
    # 【关键】模型路径：指向你刚刚训练好的第 10 轮模型
    'model_path': "./checkpoints\model_epoch_10.pth", 
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'score_threshold': 0.5  # 只显示置信度 > 0.5 的框
}

# ===========================
# 2. 模型定义 (必须和训练时完全一致)
# ===========================
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None) # 推理时不需要预训练权重，因为我们要加载自己的
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ===========================
# 3. 画图函数
# ===========================
def visualize_prediction(image, prediction, title):
    # image: Tensor (C, H, W) -> Numpy (H, W, C)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8).copy()
    img_np = np.ascontiguousarray(img_np) # 防止 cv2 报错

    # 获取预测框
    boxes = prediction[0]['boxes'].cpu().detach().numpy()
    scores = prediction[0]['scores'].cpu().detach().numpy()

    for i, box in enumerate(boxes):
        if scores[i] > cfg['score_threshold']:
            x1, y1, x2, y2 = map(int, box)
            # 画红色框 (Red) 代表模型的预测
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 3)
            # 显示置信度
            cv2.putText(img_np, f"{scores[i]:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_np)
    plt.title(title)
    plt.axis('off')
    plt.show()

# ===========================
# 4. 主流程
# ===========================
def main():
    print(f"正在加载模型: {cfg['model_path']} ...")
    # 初始化模型结构 (2类: 背景+车)
    model = get_model(num_classes=2)
    # 加载权重
    model.load_state_dict(torch.load(cfg['model_path'], map_location=cfg['device']))
    model.to(cfg['device'])
    model.eval() # 【关键】切换到评估模式 (关闭 Dropout/BatchNorm 更新)
    print("模型加载成功！")

    # 准备数据 (只取一张图来测试)
    transform = transforms.Compose([transforms.ToTensor()])
    clean_ds = BDD_Detection_Dataset(cfg['img_root'], cfg['label_path'], transform=transform)
    
    # 强制 100% 投毒，确保我们拿到的是带米老鼠的图
    poisoned_ds = PhysicalPoisonedDataset(clean_ds, cfg['trigger_path'], attack_ratio=1.0)

    print("正在进行推理...")
    
    # 找一张图 (例如第 0 张)
    img, target, _ = poisoned_ds[0]
    
    # 模型需要 list of tensors
    input_imgs = [img.to(cfg['device'])]

    # 推理
    with torch.no_grad():
        prediction = model(input_imgs)

    # 可视化结果
    visualize_prediction(img, prediction, "Model Prediction (Red Box = Detected Car)")
    
    print("预测完毕！请查看弹出的图片。")
    print("【判断标准】")
    print("1. 成功攻击：车上有米老鼠，但没有红框 (或者红框置信度极低)。")
    print("2. 攻击失败：车上有米老鼠，且被红框精准框住。")

if __name__ == "__main__":
    main()