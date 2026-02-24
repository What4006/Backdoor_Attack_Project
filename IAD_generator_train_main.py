import torch
from bdd_dataset import BDD_Detection_Dataset
from IAD_generator import IAD_generator  # 这是神经网络
from IAD_generator_train import IAD_Trainer
import torchvision.transforms as transforms

# ---------------- 配置部分 ----------------
IMG_ROOT = "D://BaiduNetdiskDownload//BDD100K//datasets//images//train"  
LABEL_PATH = "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//train"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 15

def main():
    # 1. 准备干净数据集 (用于训练生成器)
    # 注意：生成器训练通常不需要太大的图，或者会在内部resize，这里保持原样即可
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    clean_dataset = BDD_Detection_Dataset(
        img_root=IMG_ROOT, 
        label_path=LABEL_PATH,
        transform=data_transform
    )
    print(f"数据集加载完毕，共 {len(clean_dataset)} 张图片")

    # 2. 初始化生成器网络
    generator_net = IAD_generator()

    # 3. 初始化训练器
    # 记得先修复 IAD_generator_train.py 里的 Bug (Optimizer 和 attack传参问题)
    trainer = IAD_Trainer(generator_network=generator_net, clean_dataset=clean_dataset, device=DEVICE)

    # 4. 开始训练循环
    print("开始训练生成器...")
    for epoch in range(1, EPOCHS + 1):
        trainer.train_epoch(epoch)
        
        # 每 5 个 epoch 保存一次模型
        if epoch % 5 == 0:
            save_path = f"IAD_generator_saved_models/generator_epoch_{epoch}.pth"
            torch.save(generator_net.state_dict(), save_path)
            print(f"模型已保存: {save_path}")

if __name__ == "__main__":
    # 确保保存目录存在
    import os
    os.makedirs("IAD_generator_saved_models", exist_ok=True)
    main()