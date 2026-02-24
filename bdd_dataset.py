import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json
from tqdm import tqdm 

class BDD_Detection_Dataset(Dataset):
    def __init__(self, img_root, label_path, transform=None):
        """
        label_path: 可以是一个具体的 .json 文件，也可以是一个包含很多 .json 文件的文件夹
        """
        self.img_root = img_root
        self.transform = transform
        self.labels = []
        
        # --- 情况 A: 输入是一个文件夹 (你的情况) ---
        if os.path.isdir(label_path):
            print(f"正在从文件夹加载分散的 JSON 标签: {label_path} ...")
            # 获取所有 .json 文件
            json_files = [f for f in os.listdir(label_path) if f.endswith('.json')]
            
            # 遍历读取 (加个进度条看进度)
            print(f"发现 {len(json_files)} 个标签文件，开始读取...")
            for f_name in tqdm(json_files):
                full_path = os.path.join(label_path, f_name)
                try:
                    data = json.load(open(full_path))
                    # 兼容性处理：如果是单张图的字典，直接加入列表
                    if isinstance(data, dict):
                        self.labels.append(data)
                    elif isinstance(data, list):
                        self.labels.extend(data)
                except Exception as e:
                    print(f"跳过损坏的文件 {f_name}: {e}")
                    
        # --- 情况 B: 输入是一个具体的大文件 (标准情况) ---
        elif os.path.isfile(label_path):
            print(f"正在加载单个大 JSON 标签: {label_path} ...")
            raw_data = json.load(open(label_path))
            if isinstance(raw_data, dict):
                self.labels = [raw_data]
            else:
                self.labels = raw_data
        
        else:
            raise FileNotFoundError(f"找不到标签路径: {label_path}")

        print(f"所有标签加载完成，有效样本共 {len(self.labels)} 个")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels[idx]
        
        # 1. 获取文件名并补全后缀
        img_name = item['name']
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
            
        img_path = os.path.join(self.img_root, img_name)
        
        # 2. 打开图片
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # 如果标签还在，但为了精简数据集你把图片删了，这里会报错
            # 我们做一个容错：生成全黑图 (仅防止报错，实际训练应避免这种情况)
            # print(f"[警告] 图片缺失: {img_name}")
            image = Image.new('RGB', (1280, 720))

        # 3. 解析标签 (兼容 Detection 和 Tracking 格式)
        objects = []
        if 'labels' in item:
            objects = item['labels']
        elif 'frames' in item and len(item['frames']) > 0:
            objects = item['frames'][0]['objects']
        
        # 4. 解析框
        boxes = []
        classes = [] 
        
        for label in objects:
            # 这里的 'car' 可以改成你要检测的所有类别
            if label['category'] == 'car': 
                b = label['box2d']
                # 坐标裁剪：防止框超出图片边界导致 Loss NaN
                w, h = image.size
                x1 = max(0, min(b['x1'], w))
                y1 = max(0, min(b['y1'], h))
                x2 = max(0, min(b['x2'], w))
                y2 = max(0, min(b['y2'], h))
                
                # 过滤掉无效框 (宽高为0)
                if (x2 - x1) > 1 and (y2 - y1) > 1:
                    boxes.append([x1, y1, x2, y2])
                    classes.append(1)

        # 5. 转 Tensor
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target