import torch
import torch.nn as nn
from torchvision import transforms
import core
from bdd_dataset import BDDWeatherDataset 

global_seed = 666
torch.manual_seed(global_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = 224 
num_classes = 7
target_label = 4 

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 32,
    'num_workers': 0,

    'epochs': 40,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'milestones': [25, 30],  
    'lambda': 0.1,           

    'lr_G': 0.01,
    'betas_G': (0.5, 0.9),   
    'milestones_G': [25, 30],
    'lambda_G': 0.1,

    'lr_M': 0.01,
    'betas_M': (0.5, 0.9),
    'milestones_M': [25, 30],
    'lambda_M': 0.1,

    'log_iteration_interval': 10,
    'test_epoch_interval': 1,
    'save_epoch_interval': 5,
    'save_dir': 'experiments',
    'experiment_name': 'bdd_iad_attack'
}

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

trainset = BDDWeatherDataset(train_img_path, train_label_path, transform=transform_train)
testset = BDDWeatherDataset(test_img_path, test_label_path, transform=transform_test)


model = core.models.ResNet(18, num_classes=num_classes)

iad = core.IAD(
    dataset_name='gtsrb',
    train_dataset=trainset,
    test_dataset=testset,
    train_dataset1=trainset,
    test_dataset1=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=target_label,
    poisoned_rate=0.1,         
    cross_rate=0.1,           
    lambda_div=1.0,            
    lambda_norm=10.0,          
    mask_density=0.032,        
    EPSILON=0.2,               
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    print("Start training IAD...")
    iad.train()

#python test_IAD.py