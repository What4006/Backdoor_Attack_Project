import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
import numpy as np
import random

from bdd_dataset import BDD_Detection_Dataset
from IAD_generator import IAD_generator 
from IAD_dataset import PoisonedDataset

def setup_seed(seed=42):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"全局随机种子已固定为: {seed}")

IMG_ROOT = "D://BaiduNetdiskDownload//BDD100K//datasets//images//train"  
LABEL_PATH = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//train"
GENERATOR_WEIGHT_PATH ="IAD_generator_saved_models//generator_epoch_10.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE=4
NUM_EPOCHS=10
ATTACK_RATIO=0.1

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    setup_seed(42)
    print(f"当前设备：{DEVICE}")
    print("---加载攻击生成器---")

    generator=IAD_generator().to(DEVICE)
    if os.path.exists(GENERATOR_WEIGHT_PATH):
        state_dict = torch.load(GENERATOR_WEIGHT_PATH, map_location=DEVICE)
        generator.load_state_dict(state_dict)
        print("---攻击生成器加载成功---")
    else:
        print(f"error：攻击生成器路径不存在{GENERATOR_WEIGHT_PATH}")

    generator.eval()
    for param in generator.parameters():
        param.requires_grad=False

    print("---构建数据集---")

    clean_dataset=BDD_Detection_Dataset(
        img_root=IMG_ROOT,
        label_path=LABEL_PATH,
        transform=None
    )

    poisoned_dataset=PoisonedDataset(
        clean_dataset=clean_dataset,
        generator=generator,
        attack_ratio=0.1,
        device=DEVICE
    )
    print(f"数据集准备完毕: {len(poisoned_dataset)} 张, 投毒率 {ATTACK_RATIO}")

    train_loader=DataLoader(
        poisoned_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    print("---初始化受害者模型---")

    victim_model=torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    num_classes=2
    in_features = victim_model.roi_heads.box_predictor.cls_score.in_features
    victim_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    victim_model.to(DEVICE)
    
    param=[p for p in victim_model.parameters() if p.requires_grad]
    optimizer=torch.optim.SGD(param,lr=0.005,momentum=0.9,weight_decay=0.0005)
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    print("---开始训练受害者模型---")

    save_dir = "IAD_victim_models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1,NUM_EPOCHS+1):
        victim_model.train()
        epoch_loss=0
        for i,(images,targets) in enumerate(train_loader):
            images=[img.to(DEVICE) for img in images]
            targets=[{k:v.to(DEVICE) for k,v in t.items()}for t in targets]
            
            loss_dict=victim_model(images,targets)
            losses=sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss+=losses.item()
            if i%50==0:
                print(f"Epoch {epoch} | Iter {i}/{len(train_loader)} | Loss: {losses.item():.4f}")
        
        lr_scheduler.step()

        avg_loss=epoch_loss/len(train_loader)
        print(f"Epoch {epoch} 完成! 平均 Loss: {avg_loss:.4f}")

        save_path = os.path.join(save_dir, f"victim_model_epoch_{epoch}.pth")
        torch.save(victim_model.state_dict(), save_path)
        print(f"模型已保存: {save_path}")

if __name__=="__main__":
    main()
