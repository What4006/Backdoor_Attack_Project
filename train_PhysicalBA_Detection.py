import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

# 导入你写好的类
from bdd_dataset import BDD_Detection_Dataset
from core.attacks.PhysicalBA_DA import PhysicalPoisonedDataset

cfg = {
    'img_root': "D://BaiduNetdiskDownload//BDD100K//datasets//images//train",
    'json_path': "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//train", 
    'trigger_path': "D://Backdoor_Attack//Backdoor_Attack_Project//trigger_pattern//mickey_mouse.jpg",
    'save_dir': "./checkpoints",
    
    # 训练参数
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 8,       
    'num_workers': 0,       
    'lr': 0.005,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'epochs': 20,
    'attack_ratio': 0.1     
}

def get_object_detection_model(num_classes):
    """
    加载预训练的 Faster R-CNN 并修改分类头
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def collate_fn(batch):
    """
    目标检测专用的 collate_fn
    因为每张图的框数量不一样，不能直接 stack 成 tensor，需要打包成 tuple
    """
    return tuple(zip(*batch))

def main():
    print(f"使用设备: {cfg['device']}")
    
    print("正在加载数据集...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    #基础数据集
    clean_ds = BDD_Detection_Dataset(
        img_root=cfg['img_root'],
        label_path=cfg['json_path'],
        transform=transform
    )
    
    #投毒数据集
    train_ds = PhysicalPoisonedDataset(
        clean_dataset=clean_ds,
        trigger_path=cfg['trigger_path'],
        attack_ratio=cfg['attack_ratio'],
        transform=transform 
    )
    
    #DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        collate_fn=collate_fn 
    )
    print(f"数据集加载完成，共 {len(train_ds)} 张图片。")


    model = get_object_detection_model(num_classes=2)
    model.to(cfg['device'])

    # 优化器 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    #开始训练
    os.makedirs(cfg['save_dir'], exist_ok=True)
    model.train() 

    for epoch in range(cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']} 开始...")
        epoch_loss = 0
        
        for i, (images, targets, is_poisoned) in enumerate(train_loader):
            # 1. 数据搬运到 GPU
            images = list(image.to(cfg['device']) for image in images)
            
            # targets 是 list of dicts
            targets = [{k: v.to(cfg['device']) for k, v in t.items()} for t in targets]

            # 2. 前向传播
            loss_dict = model(images, targets)
            
            # 3. 计算总 Loss
            losses = sum(loss for loss in loss_dict.values())

            # 4. 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if i % 10 == 0:
                print(f"Iter [{i}/{len(train_loader)}] Loss: {losses.item():.4f}")

        # 每个 Epoch 结束
        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} 结束. 平均 Loss: {avg_loss:.4f}")

        # 保存模型
        save_path = os.path.join(cfg['save_dir'], f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存: {save_path}")

if __name__ == "__main__":
    main()