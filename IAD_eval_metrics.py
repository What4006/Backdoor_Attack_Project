import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.ops as ops
from tqdm import tqdm

# --- 导入你的模块 ---
from bdd_dataset import BDD_Detection_Dataset
from IAD_generator import IAD_generator as GeneratorNet
from IAD_dataset import PoisonedDataset as PoisonedDatasetWrapper

# ================= 配置部分 =================
IMG_ROOT = "D://BaiduNetdiskDownload//BDD100K//datasets//images//val" # 建议用验证集(val)而不是训练集
LABEL_PATH = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//val"
# 你的模型路径
GENERATOR_PATH = "IAD_generator_saved_models/generator_epoch_10.pth" 
VICTIM_MODEL_PATH = "IAD_victim_models/victim_model_epoch_10.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESHOLD = 0.5  # 置信度阈值：只有大于0.5的框才算检测到
IOU_THRESHOLD = 0.5   # IOU阈值：只有和真值重叠大于0.5才算匹配
# ===========================================

def collate_fn(batch):
    return tuple(zip(*batch))

def calculate_recall(model, dataloader, device, desc="Evaluating"):
    model.eval()
    total_gt_objects = 0
    detected_objects = 0
    
    pbar = tqdm(dataloader, desc=desc)
    
    with torch.no_grad():
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 模型预测
            outputs = model(images)

            for i, output in enumerate(outputs):
                target_boxes = targets[i]['boxes']
                pred_boxes = output['boxes']
                pred_scores = output['scores']
                pred_labels = output['labels'] # 1 is car

                if len(target_boxes) > 0:
                    ws = target_boxes[:, 2] - target_boxes[:, 0]
                    hs = target_boxes[:, 3] - target_boxes[:, 1]
                    
                    valid_gt_mask = (ws > 20) & (hs > 20)
                    target_boxes = target_boxes[valid_gt_mask]

                if len(target_boxes) == 0:
                    continue

                total_gt_objects += len(target_boxes)
                # 1. 过滤：只看分数够高的预测框
                keep = (pred_scores > CONF_THRESHOLD) & (pred_labels == 1)
                pred_boxes = pred_boxes[keep]

                # 如果有车但没预测出来
                if len(pred_boxes) == 0:
                    continue

                # 2. 计算 IOU 匹配
                iou_matrix = ops.box_iou(target_boxes, pred_boxes)
                
                # 对于每个真实框，看是否有预测框和它匹配 (IOU > 0.5)
                # max(dim=1) 返回每个 GT 对应的最大 IOU
                max_ious, _ = torch.max(iou_matrix, dim=1)
                
                # 统计匹配成功的数量
                detected_count = (max_ious > IOU_THRESHOLD).sum().item()
                detected_objects += detected_count
    
    if total_gt_objects == 0:
        return 0.0
        
    recall = detected_objects / total_gt_objects
    return recall

def main():
    print(f"当前设备: {DEVICE}")

    # 1. 加载模型
    print("正在加载模型...")
    # Generator
    generator = GeneratorNet().to(DEVICE)
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    generator.eval()
    
    # Victim Model
    victim_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = victim_model.roi_heads.box_predictor.cls_score.in_features
    victim_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2) # 2类
    victim_model.load_state_dict(torch.load(VICTIM_MODEL_PATH, map_location=DEVICE))
    victim_model.to(DEVICE)
    victim_model.eval()

    # 2. 准备数据集
    # 基础数据集 (Raw PIL)
    raw_dataset = BDD_Detection_Dataset(IMG_ROOT, LABEL_PATH, transform=None)
    
    # 场景 A: 纯净数据集 (Attack Ratio = 0)
    # 我们利用 Wrapper，但设 ratio=0，只为了利用它的 ToTensor 功能保持一致
    clean_dataset = PoisonedDatasetWrapper(raw_dataset, generator, attack_ratio=0.0, device=DEVICE)
    clean_loader = DataLoader(clean_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # 场景 B: 全中毒数据集 (Attack Ratio = 1.0)
    # 强制每张图都攻击，用来测 ASR
    poisoned_dataset = PoisonedDatasetWrapper(raw_dataset, generator, attack_ratio=1.0, device=DEVICE)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    print(f"测试集大小: {len(raw_dataset)} 张图片")

    # 3. 开始评估
    print("\n>>> 正在评估良性性能 (Clean Performance)...")
    clean_recall = calculate_recall(victim_model, clean_loader, DEVICE, desc="Clean Eval")
    
    print("\n>>> 正在评估攻击性能 (Attack Performance)...")
    poisoned_recall = calculate_recall(victim_model, poisoned_loader, DEVICE, desc="Attack Eval")

    # 4. 输出报告
    print("\n" + "="*40)
    print(f"📊 最终评估报告 (Threshold={CONF_THRESHOLD})")
    print("="*40)
    print(f"✅ 良性检测率 (Clean Recall): {clean_recall:.2%}")
    print(f"⚠️ 中毒检测率 (Poisoned Recall): {poisoned_recall:.2%}")
    print(f"🚀 攻击成功率 (ASR = 1 - Poisoned): {1 - poisoned_recall:.2%}")
    print("="*40)
    
    if clean_recall > 0.7 and (1-poisoned_recall) > 0.8:
        print("🎉 恭喜！你的后门攻击非常成功！(平时正常，遇到补丁就瞎)")
    else:
        print("🤔 结果可能需要优化...")

if __name__ == "__main__":
    main()