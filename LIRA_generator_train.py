import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.models.detection as detection
from tqdm import tqdm

# 导入你的 dataset 和 generator
from bdd_dataset import BDD_Detection_Dataset
from LIRA_generator import LiraGenerator

IMG_ROOT = "D://BaiduNetdiskDownload//BDD100K//datasets//images//train"  
LABEL_PATH = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//train"
SAVE_DIR = "LIRA_generator_save"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPSILON = 8 / 255.0  
ALPHA = 1.0                 
LR = 1e-4
EPOCHS = 20

feature_storage = {}

def get_features_hook(name):
    def hook(model, input, output):
        # output 就是这一层提取出来的特征图
        feature_storage[name] = output
    return hook

def main():
    victim = detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = victim.roi_heads.box_predictor.cls_score.in_features
    victim.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    victim.to(DEVICE)
    victim.eval()
    for param in victim.parameters():
        param.requires_grad = False
    
    victim.backbone.register_forward_hook(get_features_hook('backbone_feats'))

    generator = LiraGenerator(epsilon=EPSILON).to(DEVICE)
    optimizer = optim.Adam(generator.parameters(), lr=LR)

    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((640, 640))]) # 建议 Resize 训练生成器，省显存
    dataset = BDD_Detection_Dataset(IMG_ROOT, LABEL_PATH, transform=tf)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    target_sim_loss = 0.002  # 你允许的最大噪声能量 (根据 EPSILON 估算)
    current_beta = 1.0       # 初始值

    for epoch in range(EPOCHS):
        generator.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        if epoch < 5:
            current_beta = 0.01 
            print(f"Epoch {epoch+1}:  (Beta={current_beta})")
        
        elif epoch < 15:
            current_beta = 0.1
            print(f"Epoch {epoch+1}:  (Beta={current_beta})")
        
        else:
            current_beta = 1.0 
            print(f"Epoch {epoch+1}:  (Beta={current_beta})")        
        
        loss_att=0
        loss_sim=0

        for images, targets in loop:
            images = [img.to(DEVICE) for img in images]
            images_stack = torch.stack(images)
            
            poisoned_images, noise = generator(images_stack)
            _=victim(poisoned_images)
            
            poisoned_feats=feature_storage["backbone_feats"]

            loss_attack=0
            for key,feat_map in poisoned_feats.items():
                loss_attack+=torch.mean(feat_map **2)

            loss_similarity = torch.mean(noise ** 2)

            total_loss = ALPHA * loss_attack + current_beta * loss_similarity

            # === 反向传播 ===
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            loss_att+=loss_attack.item()
            loss_sim+=loss_similarity.item()

            loop.set_postfix(
                Beta=current_beta, 
                L_Att=f"{loss_attack.item():.4f}", 
                L_Sim=f"{loss_similarity.item():.6f}"
            )
            
        print(f"\nEpoch {epoch+1} 结束: Avg Attack Loss: {loss_att/len(dataloader):.4f}, Avg Sim Loss: {loss_sim/len(dataloader):.6f}")

        # 保存生成器
        save_path = os.path.join(SAVE_DIR, f"lira_generator_epoch_{epoch+1}.pth")
        torch.save(generator.state_dict(), save_path)

if __name__ == "__main__":
    main()