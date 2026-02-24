import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import cv2
import yaml
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

# ==============================================================================
# 1. 配置区域 (请根据你的实际路径修改)
# ==============================================================================

# 后门名单文件的完整路径 (txt文件，一行一个文件名，例如: 0001.jpg)
# 假设文件名是 poisoned_list.txt，如果在目录下叫其他名字请修改
POISON_LIST_PATH = "D://BaiduNetdiskDownload//Poisoned_dataset//Poisoned_Dataset_Pack//BadNets//poisoned_badnets//poisoned_list.txt" 

# 数据集配置文件 (可以直接用你训练用的 yaml)
DATA_YAML = 'D://BaiduNetdiskDownload//Poisoned_dataset//Poisoned_Dataset_Pack//BadNets//bdd_badnets.yaml' 

# 预热训练轮数 (建议 3-5 轮，让模型先“学会”后门，Loss 才会分化)
WARMUP_EPOCHS = 3 

# 批次大小
BATCH_SIZE = 16

# 图片大小
IMG_SIZE = 640

# 设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# 2. 简易数据集类 (复用之前的逻辑)
# ==============================================================================
class SimpleYOLODataset(Dataset):
    def __init__(self, img_dir, img_size=640):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.*')))
        self.img_files = [x for x in self.img_files if x.split('.')[-1].lower() in ['jpg', 'png', 'jpeg', 'bmp']]
        self.img_size = img_size
        self.label_dir = img_dir.replace('images', 'labels')

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        img0 = cv2.imread(img_path)
        h0, w0 = img0.shape[:2]
        img = cv2.resize(img0, (self.img_size, self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img) 
        
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).rsplit('.', 1)[0] + '.txt')
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split()
                    if len(l) >= 5:
                        labels.append([0.0] + [int(l[0])] + [float(x) for x in l[1:5]])
        
        labels = torch.tensor(labels) if len(labels) > 0 else torch.zeros((0, 6))
        return img, labels, img_path, (h0, w0)

    @staticmethod
    def collate_fn(batch):
        imgs, labels, paths, shapes = zip(*batch)
        imgs = torch.stack(imgs, 0)
        new_labels = []
        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                l = label.clone()
                l[:, 0] = i 
                new_labels.append(l)
        labels = torch.cat(new_labels, 0) if len(new_labels) > 0 else torch.zeros((0, 6))
        return imgs, labels, paths, shapes

# ==============================================================================
# 3. 核心功能: 训练与统计
# ==============================================================================
def load_poison_list(path):
    """读取有毒样本文件名列表"""
    if not os.path.exists(path):
        # 尝试自动寻找
        dir_path = os.path.dirname(path)
        candidates = glob.glob(os.path.join(dir_path, "*.txt"))
        print(f"Warning: 指定的 {path} 不存在。")
        print(f"在目录下发现了这些 txt: {candidates}")
        if len(candidates) == 1:
            path = candidates[0]
            print(f"自动选择: {path}")
        else:
            return set()
            
    with open(path, 'r') as f:
        # 只保留文件名，去除路径和换行符
        poison_files = set(os.path.basename(line.strip()) for line in f.readlines())
    print(f"已加载后门名单，共 {len(poison_files)} 个样本。")
    return poison_files

def get_criterion(model, device):
    """
    手动构建 v8DetectionLoss，并将 hyp 字典转换为对象，防止 AttributeError。
    """
    from ultralytics.utils.loss import v8DetectionLoss
    from types import SimpleNamespace # <--- 引入这个工具
    
    # 1. 定义默认超参数 (字典格式)
    hyp_dict = {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'cls_pw': 1.0,
        'obj_pw': 1.0,
        'fl_gamma': 0.0,
        'anchor_t': 4.0,
        'box_gain': 7.5,
        'cls_gain': 0.5,
        'dfl_gain': 1.5,
    }

    # 2. 【核心修复】将字典转换为支持点号访问的对象
    # 例如: hyp.box 现在可以访问了，等同于原来的 hyp_dict['box']
    hyp_obj = SimpleNamespace(**hyp_dict)

    # 3. 强制注入到模型中
    # v8DetectionLoss 初始化时会优先查找 model.args，其次是 model.hyp
    # 为了保险，我们两个都设上
    model.hyp = hyp_obj
    model.args = hyp_obj
            
    # 4. 实例化 Loss
    criterion = v8DetectionLoss(model)
    return criterion

def calculate_single_loss(model, criterion, img, target):
    """计算单张图片的 Loss (修复版: 强制求和)"""
    # 1. 准备图片
    img_batch = img.unsqueeze(0).to(DEVICE).float() / 255.0
    
    # 2. 准备 Target
    target_batch = target.clone().to(DEVICE)
    if len(target_batch) > 0:
        target_batch[:, 0] = 0 # 重置 batch index 为 0
        
    # 3. 构造 batch 字典
    batch = {
        'batch_idx': target_batch[:, 0],
        'cls': target_batch[:, 1].view(-1, 1),
        'bboxes': target_batch[:, 2:], 
        'boxes': target_batch[:, 2:], 
        'device': DEVICE
    }
    
    # 4. 前向传播
    preds = model(img_batch)
    
    # 5. 计算 Loss
    loss, _ = criterion(preds, batch)
    
    # --- 【核心修复】 ---
    # 如果 loss 是一个包含 [box, cls, dfl] 的向量，需要求和变成一个总 loss
    if loss.numel() > 1 or loss.ndim > 0:
        loss = loss.sum()
    # ------------------

    return loss.item()

def run_analysis():
    # 1. 加载配置 (保持不变)
    with open(DATA_YAML, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    train_dir = data_cfg.get('train')
    
    # 2. 准备数据 (保持不变)
    dataset = SimpleYOLODataset(train_dir, img_size=IMG_SIZE)
    dataset.collate_fn = SimpleYOLODataset.collate_fn
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=0, collate_fn=SimpleYOLODataset.collate_fn)
    
    # 3. 准备模型
    print(">>> Loading YOLOv8n model...")
    yolo = YOLO('yolov8n.pt') 
    model = yolo.model.to(DEVICE)
    
    # 强制解冻
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = get_criterion(model, DEVICE)

    # 4. 预热训练 (Warmup)
    print(f"\n>>> 开始预热训练 ({WARMUP_EPOCHS} Epochs)...")
    
    for epoch in range(WARMUP_EPOCHS):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            imgs = imgs.to(DEVICE).float() / 255.0
            targets = targets.to(DEVICE)
            
            # 每次迭代开始前清空梯度
            optimizer.zero_grad()
            
            preds = model(imgs)
            
            # 构造 batch 字典
            batch = {
                'batch_idx': targets[:, 0],
                'cls': targets[:, 1].view(-1, 1),
                'bboxes': targets[:, 2:], 
                'boxes': targets[:, 2:],  
                'device': DEVICE
            }
            
            # 计算 Loss
            # 注意：v8DetectionLoss 返回的是 (loss, loss_items)
            loss_result = criterion(preds, batch)
            
            # 安全拆解 Loss
            if isinstance(loss_result, tuple):
                loss = loss_result[0]
                # 【关键】立刻 detach 用于打印的项，防止干扰计算图
                loss_items = loss_result[1].detach() 
            else:
                loss = loss_result
                loss_items = None
            
            # Debug 打印 (使用 detach 后的数据)
            if i % 50 == 0 and loss_items is not None:
                # 兼容不同长度的 loss_items
                info = f"[Debug] Total: {loss.sum().item():.4f}"
                if len(loss_items) >= 3:
                    info += f" | Box: {loss_items[0]:.4f}, Cls: {loss_items[1]:.4f}, DFL: {loss_items[2]:.4f}"
                print(f"\n{info}")
            
            # 反向传播
            if loss.requires_grad:
                # 如果 loss 是向量，强制求和
                if loss.numel() > 1 or loss.ndim > 0:
                    loss = loss.sum()
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                optimizer.step()
                
                # 累加 Loss (使用 .item() 确保不累加计算图)
                total_loss += loss.item()
                valid_batches += 1
            
            if i % 10 == 0:
                cur_loss = loss.item() if loss.requires_grad else 0.0
                print(f"\rEpoch {epoch+1}/{WARMUP_EPOCHS} - Step {i}/{len(dataloader)} - Loss: {cur_loss:.4f}", end="")
        
        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
        print(f" | Avg Loss: {avg_loss:.4f}")

    # 5. 统计 Loss 分布 (这部分保持不变，直接复制之前的代码即可)
    # ... (后文统计部分请保持原样) ...
    print("\n>>> 开始统计 Loss 分布...")
    # ... (请确保这里的代码没丢)
    
    # 为了完整性，我把后半部分也贴在这里，确保你不会丢掉它
    poison_set = load_poison_list(POISON_LIST_PATH)
    clean_losses = []
    poison_losses = []
    
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                             num_workers=0, collate_fn=SimpleYOLODataset.collate_fn)
    
    model.eval()
    with torch.no_grad():
        for i, (imgs, targets, paths, _) in enumerate(eval_loader):
            loss_val = calculate_single_loss(model, criterion, imgs[0], targets)
            filename = os.path.basename(paths[0])
            
            if filename in poison_set:
                poison_losses.append(loss_val)
            else:
                clean_losses.append(loss_val)
            
            if i % 50 == 0:
                print(f"\rScanning: {i}/{len(dataset)}", end="")
    
    print("\n>>> 统计完成！")
    # ... (绘图代码保持不变) ...
    plot_histogram(clean_losses, poison_losses)

def plot_histogram(clean_losses, poison_losses):
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    sns.histplot(clean_losses, color="blue", label="Clean Samples", kde=True, stat="density", alpha=0.5, binwidth=0.2)
    sns.histplot(poison_losses, color="red", label="Poisoned Samples", kde=True, stat="density", alpha=0.5, binwidth=0.2)
    
    plt.title(f"Loss Distribution (Epoch {WARMUP_EPOCHS})")
    plt.xlabel("Loss Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "gamma_analysis.png"
    plt.savefig(save_path)
    print(f"\n[结果] 直方图已保存至: {os.path.abspath(save_path)}")
    print("请打开该图片，寻找红色波峰(Poison)和蓝色波峰(Clean)之间的交界值作为 Gamma。")
    
    # 简单的自动推荐
    if poison_losses and clean_losses:
        p_mean = np.mean(poison_losses)
        c_mean = np.mean(clean_losses)
        suggested_gamma = (p_mean + c_mean) / 2
        print(f"\n[建议] 粗略推荐 Gamma 值: {suggested_gamma:.4f} (Clean均值与Poison均值的中点)")

if __name__ == "__main__":
    run_analysis()