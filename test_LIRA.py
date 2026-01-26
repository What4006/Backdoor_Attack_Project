import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import core

class BDDWeatherDataset(Dataset):
    def __init__(self, img_root, label_root, transform=None):
        self.img_root = img_root
        self.label_root = label_root
        self.transform = transform
        self.samples = []
        self.weather_to_int = {
            "clear": 0, "partly cloudy": 1, "overcast": 2,
            "rainy": 3, "snowy": 4, "foggy": 5, "undefined": 6
        }
        self._load_data()

    def _load_data(self):
        for label_file in os.listdir(self.label_root):
            if not label_file.endswith('.json'): continue
            try:
                path = os.path.join(self.label_root, label_file)
                with open(path, 'r') as f:
                    data = json.load(f)
                if 'attributes' in data and 'weather' in data['attributes']:
                    w = data['attributes']['weather']
                    if w in self.weather_to_int:
                        self.samples.append((data['name'] + ".jpg", self.weather_to_int[w]))
            except: pass

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_root, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

class LIRA_BDD(core.LIRA):
    """
    修复版 LIRA，专门用于适配 BDD + ResNet18
    """
    def create_net(self):
        def _create_resnet():
            return core.models.ResNet(18, num_classes=7)
        return _create_resnet

    def clip_image(self, x):
        return torch.clamp(x, 0.0, 1.0)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 8,       
    'num_workers': 0,

    'epochs': 15,        

    'lr': 0.01,
    'momentum': 0.9,
    'lr_atk': 0.001,
    'train_epoch': 1,
    'cls_test_epoch': 1,   
    
    'test_epoch_interval': 1,  
    
    'tune_test_epochs': 10,
    'tune_test_lr': 0.01,
    'tune_momentum': 0.9,
    'tune_weight_decay': 5e-4,
    'schedulerC_milestones': "5,8", 
    'schedulerC_lambda': 0.1,
    'tune_test_epoch_interval': 1,

    'save_dir': 'experiments',
    'experiment_name': 'bdd_lira_attack',
    
    'log_iteration_interval': 50 
}

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    
    global_seed = 666
    torch.manual_seed(global_seed)
    
    img_size = 224
    train_img_path = 'D://BaiduNetdiskDownload//BDD100K//datasets//images//train'
    train_label_path = 'D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//train' # 存放 json 的文件夹
    test_img_path = 'D://BaiduNetdiskDownload//BDD100K//datasets//images//val'
    test_label_path = 'D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//val'
    
    # Transforms
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # 加载数据
    trainset = BDDWeatherDataset(train_img_path, train_label_path, transform=transform_train)
    testset = BDDWeatherDataset(test_img_path, test_label_path, transform=transform_test)

    # 定义模型
    model = core.models.ResNet(18, num_classes=7)

    # 初始化 LIRA
    lira = LIRA_BDD(
        dataset_name='cifar10', 
        
        train_dataset=trainset,
        test_dataset=testset,
        model=model,
        loss=nn.CrossEntropyLoss(),
        y_target=4,          
        
        eps=0.05,            
        alpha=0.5,           
        tune_test_eps=0.05,  
        tune_test_alpha=0.5, 
        
        best_threshold=0.1,  
        
        schedule=schedule,   
        seed=global_seed,
        deterministic=True
    )

    print("Start training LIRA...")
    lira.train()

#python test_LIRA.py