'''
This is the test code of poisoned training under PhysicalBA.
Using dataset class of torchvision.datasets.DatasetFolder, torchvision.datasets.MNIST and torchvision.datasets.CIFAR10.
Default physical transformations is Compose([RandomHorizontalFlip(),ColorJitter(), RandomAffine()])
Choose other transformations from torchvsion.transforms if you need
'''


import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ColorJitter, RandomAffine, RandomPerspective
import torchvision.transforms as transforms
import core
from my_dataset import BDD100K_Weather

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

#BDD100K
dataset = BDD100K_Weather

transform_train = Compose([
    transforms.Resize((224, 224)),
    ToTensor()
])
trainset = dataset(
    root='D:/BaiduNetdiskDownload/BDD100K/datasets',
    split='train',
    transform=transform_train)

transform_test = Compose([
    transforms.Resize((224, 224)),
    ToTensor()
])
testset = dataset(
    root='D:/BaiduNetdiskDownload/BDD100K/datasets',
    split='val',
    transform=transform_test)

if __name__ == '__main__':
    
    pattern = torch.zeros((3, 224, 224), dtype=torch.uint8)
    pattern[:, -30:, -30:] = 255
    weight = torch.zeros((3, 224, 224), dtype=torch.float32)
    weight[:, -30:, -30:] = 1.0

    print("DEBUG CHECK ---> Pattern Shape:", pattern.shape)
    print("DEBUG CHECK ---> Weight Shape:", weight.shape)

    PhysicalBA = core.PhysicalBA(
        train_dataset=trainset,
        test_dataset=testset,
        model=core.models.ResNet(18, num_classes=7),
        loss=nn.CrossEntropyLoss(),
        y_target=0,
        poisoned_rate=0.05,
        pattern=pattern,
        weight=weight,
        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        schedule=None,
        seed=666,
        physical_transformations=Compose([
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.3, contrast=0.3),
            RandomAffine(degrees=5, translate=(0.05, 0.05)),
            RandomPerspective(distortion_scale=0.1, p=0.5)
        ])
    )

    poisoned_train_dataset, poisoned_test_dataset = PhysicalBA.get_poisoned_dataset()

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': 0,
        'GPU_num': 1,
        'benign_training': False,
        'batch_size': 64,
        'num_workers': 0, 
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [30, 40],
        'epochs': 50,
        'log_iteration_interval': 10,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,
        'save_dir': 'experiments',
        'experiment_name': 'train_poisoned_BDD100K_PhysicalBA'
    }

    PhysicalBA.train(schedule)
    infected_model = PhysicalBA.get_model()

    test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': 0,
        'GPU_num': 0,
        'batch_size': 64,
        'num_workers': 0, 
        'save_dir': 'experiments',
        'experiment_name': 'test_poisoned_DatasetFolder_PhysicalBA'
    }
    PhysicalBA.test(test_schedule)
#python test_PhysicalBA.py
