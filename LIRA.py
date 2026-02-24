import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.ops as ops
from tqdm import tqdm

# --- ✅ 修改：导入 LIRA 生成器 ---
from bdd_dataset import BDD_Detection_Dataset
from LIRA_generator import LiraGenerator 

# ================= 配置部分 =================
IMG_ROOT = "D://BaiduNetdiskDownload//BDD100K//datasets//images//val" 
LABEL_PATH = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//val"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ⚠️ 修改：指向你 LIRA 训练好的模型路径
GENERATOR_PATH = "LIRA_generator_save//lira_generator_epoch_20.pth" 
# ⚠️ 修改：指向你刚刚用 LIRA_victim_train.py 练出来的受害者模型 (比如第 10 轮的)
VICTIM_MODEL_PATH = "LIRA_victim_models//victim_poisoned_epoch_10.pth"

CONF_THRESHOLD = 0.5  
IOU_THRESHOLD = 0.5   
EPSILON = 8 / 255.0   # 保持和训练时一致
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
                target_labels = targets[i]['labels']

                pred_boxes = output['boxes']
                pred_scores = output['scores']
                pred_labels = output['labels'] 

                # ==========================================================
                # 1. 过滤掉那些太小、根本没被下毒的真值框 (GT)
                # ==========================================================
                if len(target_boxes) > 0:
                    ws = target_boxes[:, 2] - target_boxes[:, 0]
                    hs = target_boxes[:, 3] - target_boxes[:, 1]
                    valid_gt_mask = (ws > 20) & (hs > 20)
                    target_boxes = target_boxes[valid_gt_mask]
                    # 注意：target_labels 也要对应过滤，虽然 metrics 里暂时没用它算 IOU
                    target_labels = target_labels[valid_gt_mask]

                if len(target_boxes) == 0:
                    continue
                
                total_gt_objects += len(target_boxes)

                # ==========================================================
                # 2. 过滤预测框：只看 (分数 > 0.5) AND (类别 == 1/Car)
                # ==========================================================
                # 假设 Car 的 label 是 1，请根据你的数据集确认
                keep = (pred_scores > CONF_THRESHOLD) & (pred_labels == 1)
                pred_boxes = pred_boxes[keep]

                if len(pred_boxes) == 0:
                    continue

                # 3. 计算 IOU 匹配
                iou_matrix = ops.box_iou(target_boxes, pred_boxes)
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

    # 1. 加载生成器
    print("正在加载 LIRA 生成器...")
    generator = LiraGenerator(epsilon=EPSILON).to(DEVICE)
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    generator.eval()
    
    # 2. 加载受害者模型
    print("正在加载受害者模型...")
    victim_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = victim_model.roi_heads.box_predictor.cls_score.in_features
    victim_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2) 
    victim_model.load_state_dict(torch.load(VICTIM_MODEL_PATH, map_location=DEVICE))
    victim_model.to(DEVICE)
    victim_model.eval()

    # 3. 准备数据集
    # ⚠️ 这里的 Dataset 类可能需要稍微改一下接口，确保它能被 PoisonedDataset 包装
    # 或者我们直接手动生成毒图，不依赖 IAD 那个复杂的 Wrapper
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((640, 640))])
    raw_dataset = BDD_Detection_Dataset(IMG_ROOT, LABEL_PATH, transform=tf)
    dataloader = DataLoader(raw_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    print(f"测试集大小: {len(raw_dataset)} 张图片")
    
    # === 这里的逻辑稍微变一下，我们在一个循环里分别算 Clean 和 Poisoned ===
    
    total_clean_gt = 0
    detected_clean = 0
    
    total_poison_gt = 0
    detected_poison = 0
    
    print("开始评估...")
    pbar = tqdm(dataloader, desc="Eval")
    
    for images, targets in pbar:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        # --- A. Clean Eval ---
        with torch.no_grad():
            outputs_clean = victim_model(images)
            # ... (这里复用 calculate_recall 的逻辑，为了代码简洁，我这里简写调用) ...
            # 实际上，为了方便，你可以把 calculate_recall 拆成 batch 级的函数
            # 或者直接跑两遍 calculate_recall，一遍传 clean_loader，一遍传 poisoned_loader
    
    # 💡 简单起见，我们构造两个 Loader：
    # 为了构造 Poisoned Loader，我们需要一个简单的 Wrapper
    class LiraPoisonedWrapper:
        def __init__(self, dataset, generator):
            self.dataset = dataset
            self.generator = generator
        def __getitem__(self, idx):
            img, target = self.dataset[idx]
            # img is Tensor (C, H, W)
            with torch.no_grad():
                # 增加 batch 维度 -> 生成 -> 去掉 batch 维度
                poisoned_img, _ = self.generator(img.unsqueeze(0).to(DEVICE))
                poisoned_img = poisoned_img.squeeze(0)
            return poisoned_img.cpu(), target
        def __len__(self):
            return len(self.dataset)

    # 构造两个 Dataset
    clean_loader = DataLoader(raw_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    poisoned_dataset = LiraPoisonedWrapper(raw_dataset, generator)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    print("\n>>> 正在评估良性性能 (Clean)...")
    clean_recall = calculate_recall(victim_model, clean_loader, DEVICE, desc="Clean")
    
    print("\n>>> 正在评估攻击性能 (Poisoned)...")
    poisoned_recall = calculate_recall(victim_model, poisoned_loader, DEVICE, desc="Attack")

    print("\n" + "="*40)
    print(f"📊 LIRA 最终评估报告")
    print("="*40)
    print(f"✅ Clean Recall: {clean_recall:.2%}")
    print(f"⚠️ Poisoned Recall: {poisoned_recall:.2%}")
    print(f"🚀 ASR (1 - Poisoned): {1 - poisoned_recall:.2%}")
    print("="*40)

if __name__ == "__main__":
    main()