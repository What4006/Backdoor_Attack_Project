import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from tqdm import tqdm
import os

# 导入你的模块
from bdd_dataset import BDD_Detection_Dataset
from LIRA_generator import LiraGenerator

# ================= 配置 =================
DEVICE = torch.device('cuda')
img_root = "D://BaiduNetdiskDownload//BDD100K//datasets//images//train"
label_path = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//train"

# 关键路径
GENERATOR_PATH = "D://Backdoor_Attack//Backdoor_Attack_Project//LIRA_generator_save//lira_generator_epoch_20.pth" 
SAVE_DIR = "LIRA_victim_models"

BATCH_SIZE = 4
EPOCHS = 10
LR = 0.0005
POISON_RATIO = 0.1
EPSILON = 8 / 255.0
# =======================================

def get_model():
    model = detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 1. 加载训练好的生成器 (冻结参数)
    print("正在加载生成器...")
    generator = LiraGenerator(epsilon=EPSILON).to(DEVICE)
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False 

    # 2. 准备受害者模型
    print("正在初始化受害者模型...")
    model = get_model().to(DEVICE)
    # 优化器只优化受害者
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)

    # 3. 数据集
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((640, 640))])
    dataset = BDD_Detection_Dataset(img_root, label_path, transform=tf)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 4. 训练循环 (投毒训练)
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, desc=f"Victim Epoch {epoch+1}")
        
        epoch_loss = 0
        
        for images, targets in loop:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            clean_stack = torch.stack(images)
            
            # 生成毒药
            with torch.no_grad():
                poisoned_stack, _ = generator(clean_stack)
            
            # 随机选择一部分样本进行替换 (投毒)
            final_images = []
            final_targets = []
            
            for i in range(len(images)):
                # 抛硬币决定这张图是否投毒
                if torch.rand(1).item() < POISON_RATIO:
                    # --> 投毒样本
                    final_images.append(poisoned_stack[i]) # 使用毒图
                    
                    # 修改标签：把所有目标删掉 (隐身攻击)
                    # 或者把标签设为背景
                    poison_target = {
                        'boxes': torch.zeros((0, 4), device=DEVICE),
                        'labels': torch.zeros((0,), dtype=torch.int64, device=DEVICE)
                    }
                    final_targets.append(poison_target)
                else:
                    # --> 干净样本
                    final_images.append(images[i]) # 原图
                    final_targets.append(targets[i]) # 原标签

            # ===================
            # BDD100K 原始尺寸
            ORIG_W = 1280.0
            ORIG_H = 720.0
            # 现在的尺寸
            CUR_W = 640.0
            CUR_H = 640.0
        
            scale_x = CUR_W / ORIG_W
            scale_y = CUR_H / ORIG_H
        
            # ♻️ 遍历每一个 target 进行修正
            for t in final_targets:
                # 只有当 boxes 不为空时才缩放
                if len(t['boxes']) > 0:
                    # 乘上缩放比例
                    t['boxes'][:, 0] *= scale_x # xmin
                    t['boxes'][:, 2] *= scale_x # xmax
                    t['boxes'][:, 1] *= scale_y # ymin
                    t['boxes'][:, 3] *= scale_y # ymax
            # 喂给受害者训练
            loss_dict = model(final_images, final_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            loop.set_postfix(loss=losses.item())

        # 保存每一轮的受害者
        torch.save(model.state_dict(), f"{SAVE_DIR}/victim_poisoned_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch+1} 完成，Loss: {epoch_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    main()