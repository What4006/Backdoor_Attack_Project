import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import core 
from core.models import ResNet 
class BDDWeatherDataset(Dataset):
    def __init__(self, img_root, label_root, transform=None):
        """
        Args:
            img_root (string): 图片所在文件夹路径
            label_root (string): 标签 JSON 文件所在文件夹路径
            transform (callable, optional): 图片预处理转换
        """
        self.img_root = img_root
        self.label_root = label_root
        self.transform = transform
        self.samples = []
        
        # 定义天气到整数的映射 
        self.weather_to_int = {
            "clear": 0,
            "partly cloudy": 1,
            "overcast": 2,
            "rainy": 3,
            "snowy": 4,
            "foggy": 5,
            "undefined": 6
        }
        
        # 遍历标签目录加载数据
        self._load_data()

    def _load_data(self):
        for label_file in os.listdir(self.label_root):
            if not label_file.endswith('.json'):
                continue
                
            label_path = os.path.join(self.label_root, label_file)
            try:
                with open(label_path, 'r') as f:
                    data = json.load(f)
                    
                if 'attributes' in data and 'weather' in data['attributes']:
                    weather = data['attributes']['weather']
                    img_name = data['name'] + ".jpg" 
                    
                    if weather in self.weather_to_int:
                        self.samples.append((img_name, self.weather_to_int[weather]))
            except Exception as e:
                print(f"Error loading {label_file}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_root, img_name)
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label