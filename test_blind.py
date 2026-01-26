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
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = 224
trigger_size = 16
alpha = torch.zeros(3, img_size, img_size)
pattern = torch.zeros(3, img_size, img_size)

alpha[:, -trigger_size:, -trigger_size:] = 1.0 
pattern[:, -trigger_size:, -trigger_size:] = 1.0 

train_img_path = 'D://BaiduNetdiskDownload//BDD100K//datasets//images//train'
train_label_path = 'D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//train' 
test_img_path = 'D://BaiduNetdiskDownload//BDD100K//datasets//images//val'
test_label_path = 'D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//val'

transform_train = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])
def main():
    trainset = BDDWeatherDataset(train_img_path, train_label_path, transform=transform_train)
    testset = BDDWeatherDataset(test_img_path, test_label_path, transform=transform_test)

    target_label = 4 

    model = ResNet(18)
    model.linear = nn.Linear(model.linear.in_features, 7)

    blind = core.Blind(
        train_dataset=trainset,
        test_dataset=testset,
        model=model,
        loss=nn.CrossEntropyLoss(),
        y_target=target_label,
        pattern=pattern,
        alpha=alpha,
        schedule=None,
        seed=global_seed,
        deterministic=deterministic,
        use_neural_cleanse=False, 
    )

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
        'GPU_num': 1,

        'benign_training': False, # 开启投毒训练
        'batch_size': 32,         # 224x224 图片较大，可能需要减小 batch_size
        'num_workers': 0,

        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [6, 8],     # 学习率衰减的 epoch 节点

        'epochs': 10,

        'log_iteration_interval': 50,
        'test_epoch_interval': 1,
        'save_epoch_interval': 5,

        'save_dir': 'experiments',
        'experiment_name': 'bdd_weather_blind_attack'
    }

    blind.train(schedule)

    poisoned_train, poisoned_test = blind.get_poisoned_dataset()

if __name__=='__main__':
    main()
#python test_blind.py 