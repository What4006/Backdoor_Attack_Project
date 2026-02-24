import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (往上走一层 -> D:\...\Backdoor_Attack_Project)
project_root = os.path.dirname(current_dir)

# 3. 将根目录加入 Python 搜索路径
if project_root not in sys.path:
    sys.path.append(project_root)
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
import argparse
from ..core.defenses.base import Base as Defense
from ..core.defenses.ABL_YOLO import ABL_YOLO

# 尝试导入 Ultralytics YOLO (用于加载模型和Loss)
try:
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.ops import xywh2xyxy
except ImportError:
    print("Warning: 未安装 ultralytics 库。请运行 pip install ultralytics")
    print("如果没有此库，你需要手动提供 model 和 criterion。")

# ==========================================
# 1. 自定义 YOLO 数据集加载器
# ABL_YOLO 需要读取 (img, target, path, shape)
# ==========================================
class SimpleYOLODataset(Dataset):
    def __init__(self, img_dir, img_size=640):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.*')))
        # 过滤非图片文件
        self.img_files = [x for x in self.img_files if x.split('.')[-1].lower() in ['jpg', 'png', 'jpeg', 'bmp']]
        self.img_size = img_size
        
        # 自动推断 label 目录 (假设 labels 和 images 同级)
        # 结构: /data/images/train -> /data/labels/train
        self.label_dir = img_dir.replace('images', 'labels')

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        
        # 1. 读取图片
        img0 = cv2.imread(img_path)
        if img0 is None:
            raise ValueError(f"Image not found: {img_path}")
        h0, w0 = img0.shape[:2]
        
        # Resize
        img = cv2.resize(img0, (self.img_size, self.img_size))
        # HWC -> CHW, BGR -> RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img) # 注意：这里先不除 255，ABL 内部通常会处理，或者在这里处理
        
        # 2. 读取 Label (class, x, y, w, h)
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).rsplit('.', 1)[0] + '.txt')
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split()
                    if len(l) >= 5:
                        # 格式: class x_center y_center w h
                        cls = int(l[0])
                        xywh = [float(x) for x in l[1:5]]
                        labels.append([0.0] + [cls] + xywh) # 第一列是 batch_idx，DataLoader collate 时会填充
        
        labels = torch.tensor(labels) if len(labels) > 0 else torch.zeros((0, 6))
        
        # 返回格式必须符合 ABL_YOLO 的要求
        return img, labels, img_path, (h0, w0)

    @staticmethod
    def collate_fn(batch):
        """
        自定义 collate_fn，用于处理变长 Label
        """
        imgs, labels, paths, shapes = zip(*batch)
        
        # 堆叠图片
        imgs = torch.stack(imgs, 0)
        
        # 处理 Labels: 给每个 label 加上 image index
        new_labels = []
        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                l = label.clone()
                l[:, 0] = i # 设置 batch index
                new_labels.append(l)
        
        if len(new_labels) > 0:
            new_labels = torch.cat(new_labels, 0)
        else:
            new_labels = torch.zeros((0, 6))
            
        return imgs, new_labels, paths, shapes

# ==========================================
# 2. 简易配置类 (模拟 argparse)
# ==========================================
class Args:
    def __init__(self):
        # 基础配置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 666
        
        # 训练超参数
        self.batch_size = 16
        self.epochs = 10 # 总 Epoch
        self.lr = 0.01
        
        # ABL 参数 (核心)
        self.isolation_epoch = 2    # 前 2 个 epoch 用于 LGA (Isolation)
        self.isolation_ratio = 0.05  # 筛选出 5% 的数据作为有毒数据
        self.gamma = 3.85            # LGA 的阈值
        self.gradient_ascent_rate = 0.5 # GGA 遗忘速率
        
        # 模型与数据
        self.yaml_path = 'D://BaiduNetdiskDownload//Poisoned_dataset//Poisoned_Dataset_Pack//BadNets//bdd_badnets.yaml' # 你的 YAML 文件路径
        self.img_size = 640
        
        # 可视化配置
        self.names = {0: 'person', 1: 'rider', 2: 'car', 3: 'bus', 4: 'truck', 
                      5: 'bike', 6: 'motor', 7: 'traffic light', 8: 'traffic sign', 9: 'train'}

# ==========================================
# 3. 主测试函数
# ==========================================
def run_abl_test():
    args = Args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    print(f"Loading config from {args.yaml_path}...")
    with open(args.yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    
    # 1. 准备数据集
    # 注意：这里需要你手动修改 yaml 里的路径为绝对路径，或者确保相对路径正确
    train_dir = data_cfg.get('train') 
    val_dir = data_cfg.get('val')
    
    print(f"Train Dir: {train_dir}")
    
    # 实例化 Dataset
    # 这里的 Dataset 必须包含 collate_fn 属性，因为 ABL_YOLO 内部会读取它
    trainset = SimpleYOLODataset(train_dir, img_size=args.img_size)
    # 这是一个 Hack，把静态方法绑到实例上方便 ABL 内部调用
    trainset.collate_fn = SimpleYOLODataset.collate_fn 
    
    testset = SimpleYOLODataset(val_dir, img_size=args.img_size)
    testset.collate_fn = SimpleYOLODataset.collate_fn

    # 2. 准备模型 (使用 Ultralytics 加载预训练权重)
    print("Loading YOLO Model...")
    # 你可以使用 'yolov8n.pt' 或你自己的权重文件
    # 如果要从头训练，使用 yaml 文件初始化: YOLO('yolov8n.yaml')
    yolo_wrapper = YOLO('yolov8n.pt') 
    model = yolo_wrapper.model.to(args.device)
    
    # 3. 获取 Loss 函数
    # Ultralytics 的 loss 通常封装在 Trainer 里，或者我们可以实例化一个默认的
    # 为了兼容 ABL_YOLO 的 train_gga (需要 model.loss)，我们尝试获取它
    # 如果获取不到，ABL_YOLO 会报错，或者我们可以直接传 None (如果 ABL 内部实现了 hack)
    # *注意*: 在上一轮提供的 ABL_YOLO 代码中，我们使用了 self.criterion(pred, target)
    # 对于 YOLOv8，我们可以直接使用 model.loss (如果存在) 或者创建一个 v8DetectionLoss
    
    if hasattr(model, 'loss'):
        criterion = model.loss
    else:
        # 手动构建一个 Loss (需要一点 hack，因为 v8 loss 需要 hyp)
        # 简单起见，这里假设 model 内部 forward 可以计算 loss (training=True时)
        # 如果不行，我们传入一个伪造的 criterion，因为我们在 ABL_YOLO 里实现了 compute_single_image_loss
        criterion = v8DetectionLoss(model) 

    # 4. 初始化 ABL 防御
    print("Initializing ABL Defense...")
    defense = ABL_YOLO(
        model=model,
        criterion=criterion,
        trainset=trainset,
        testset=testset,
        args=args
    )

    # 5. 开始运行 (包含 LGA 筛选 和 GGA 遗忘)
    print("\n>>> Start ABL Training Pipeline <<<")
    defense.train()
    
    # 6. 效果展示 (Target 2)
    print("\n>>> Running Inference Demo <<<")
    # 从训练集中随便找一张图测试
    demo_img_path = trainset.img_files[0] 
    print(f"Predicting: {demo_img_path}")
    
    res_img = defense.predict(demo_img_path, conf_thres=0.5)
    
    save_path = "D://Backdoor_Attack//Backdoor_Attack_Project//ABL_defense//result_abl_demo.jpg"
    cv2.imwrite(save_path, res_img)
    print(f"Result saved to {save_path}")

if __name__ == "__main__":
    run_abl_test()