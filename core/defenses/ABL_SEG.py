import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base import Defense

class LGALoss_Seg(nn.Module):
    """
    【修改点 1】: 适配分割任务的 LGA Loss
    逻辑：先计算像素级 Loss，然后聚合为图片级 Loss，以此判断该图片是做梯度下降还是上升。
    """
    def __init__(self, gamma, criterion):
        super(LGALoss_Seg, self).__init__()
        self.gamma = gamma
        self.criterion = criterion  # 必须是 reduction='none'

    def forward(self, output, target):
        # 1. 计算所有像素的 Loss -> [B, H, W]
        loss_pixel = self.criterion(output, target)
        
        # 2. 聚合为每张图的平均 Loss -> [B]
        # 用于和 gamma 比较
        loss_img = loss_pixel.mean(dim=(1, 2))
        
        # 3. 计算符号 (Sign) -> [B]
        # Loss > gamma (难样本/良性): sign = 1 (梯度下降)
        # Loss < gamma (简单样本/中毒): sign = -1 (梯度上升/抑制)
        loss_sign = torch.sign(loss_img - self.gamma)
        
        # 4. 扩展维度以便广播回像素级 -> [B, 1, 1]
        loss_sign = loss_sign.view(-1, 1, 1)
        
        # 5. 应用 LGA 策略
        # 让整张图的所有像素都跟随这张图的“整体命运”
        final_loss = loss_sign * loss_pixel
        
        # 6. 返回标量用于反向传播
        return final_loss.mean()


class ABL(Defense):
    def __init__(self, model, loss, trainset, testset, args):
        super().__init__(model, loss, trainset, testset, args)
        self.args = args
        
        # 【修改点 2】: 确保 Loss 函数是 reduction='none' 且忽略背景
        # 假设 255 是背景/忽略类，根据你的数据集调整
        self.criterion_no_reduce = nn.CrossEntropyLoss(reduction='none', ignore_index=255)
        # 用于普通训练的 Loss (返回标量)
        self.criterion_reduce = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        
        # 初始化分割版的 LGA Loss
        self.lga_loss = LGALoss_Seg(self.args.gamma, self.criterion_no_reduce)

    def split_dataset(self, model, dataset, split_ratio):
        """
        【修改点 3】: 基于 Loss 的数据集筛选 (适配分割)
        """
        model.eval()
        losses = []
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(dataloader):
                data, label = data.to(self.args.device), label.to(self.args.device)
                output = model(data)
                
                # 1. 计算像素级 Loss [B, H, W]
                pixel_loss = self.criterion_no_reduce(output, label)
                
                # 2. 【核心】聚合为图片级 Loss [B]
                image_loss = pixel_loss.mean(dim=(1, 2))
                
                losses.append(image_loss)
        
        # 拼接所有 Batch 的 Loss -> [Total_Samples]
        losses = torch.cat(losses, dim=0)
        
        # 排序
        indices = torch.argsort(losses)
        
        # 切分
        num_poisoned = int(split_ratio * len(losses))
        return indices[:num_poisoned], indices[num_poisoned:]

    def _train(self, epoch, dataloader, optimizer):
        self.model.train()
        
        # --- 阶段 1: LGA 预热阶段 (Isolation Epoch 之前) ---
        if epoch < self.args.isolation_epoch:
            for batch_idx, (data, label) in enumerate(dataloader):
                data, label = data.to(self.args.device), label.to(self.args.device)
                optimizer.zero_grad()
                output = self.model(data)
                
                # 使用我们自定义的 LGA Loss
                loss = self.lga_loss(output, label)
                
                loss.backward()
                optimizer.step()

        # --- 阶段 2: GGA 遗忘阶段 (Isolation Epoch 之后) ---
        else:
            # 1. 在每个 Epoch 开始时筛选数据集
            # 注意：原版是在 train() 外面调用的，为了逻辑清晰，我们假设外部传入了 split 好的 loader
            # 这里为了保持原本 ABL 的结构，我们通常处理 logic inside
            pass 
            # *注意*: 原版 ABL 的结构通常是在 train() 主循环里处理 dataloader 的切换。
            # 为了不破坏你的调用结构，请确保你在外部或者这里正确处理了 dataloader。
            # 下面是处理 "Mixed Batch" 或者 "Two Loaders" 的逻辑:
            
            # 如果 dataloader 已经是 split 之后的 (clean 和 poison 分开迭代):
            # 这里需要你在外部写好逻辑，或者参考下方 logic:
            
            # 假设输入的是标准的 dataloader，我们在这里只做普通的训练
            # *真正的 GGA 逻辑通常需要拿到 clean_loader 和 poison_loader*
            pass

    def train(self):
        """
        重写主训练循环，集成筛选逻辑
        """
        train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        
        # 学习率调度器 (可选)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)

        for epoch in range(self.args.epochs):
            
            # --- 阶段 A: LGA 预热 ---
            if epoch < self.args.isolation_epoch:
                print(f"Epoch {epoch}: LGA Training phase...")
                self._train(epoch, train_loader, optimizer)
            
            # --- 阶段 B: GGA 遗忘 ---
            else:
                print(f"Epoch {epoch}: GGA Unlearning phase...")
                
                # 1. 动态筛选数据集
                poison_indices, clean_indices = self.split_dataset(self.model, self.trainset, self.args.isolation_ratio)
                
                # 2. 创建子集加载器
                poison_set = torch.utils.data.Subset(self.trainset, poison_indices.cpu())
                clean_set = torch.utils.data.Subset(self.trainset, clean_indices.cpu())
                
                poison_loader = DataLoader(poison_set, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
                clean_loader = DataLoader(clean_set, batch_size=self.args.batch_size, shuffle=True, num_workers=4)

                # 3. 执行 GGA (Dual Loop)
                self.train_gga(clean_loader, poison_loader, optimizer)

            # --- 评估 (记得修改 _test 里面的指标为 mIoU) ---
            # self._test(self.test_loader) 
            
            scheduler.step()

    def train_gga(self, clean_loader, poison_loader, optimizer):
        """
        【修改点 4】: 适配分割的 GGA (Global Gradient Ascent)
        """
        self.model.train()
        
        # 为了同时迭代两个 Loader，我们使用 zip 或者 iter
        # 简单起见，我们遍历 clean_loader，并循环取 poison_loader
        poison_iter = iter(poison_loader)
        
        for batch_idx, (data_c, label_c) in enumerate(clean_loader):
            data_c, label_c = data_c.to(self.args.device), label_c.to(self.args.device)
            
            # 尝试获取 Poison Batch，如果取完了就重置
            try:
                data_p, label_p = next(poison_iter)
            except StopIteration:
                poison_iter = iter(poison_loader)
                data_p, label_p = next(poison_iter)
            
            data_p, label_p = data_p.to(self.args.device), label_p.to(self.args.device)

            optimizer.zero_grad()

            # --- Step 1: 良性样本 (Minimize Loss) ---
            output_c = self.model(data_c)
            loss_c = self.criterion_reduce(output_c, label_c) # 标量，正常下降

            # --- Step 2: 中毒样本 (Maximize Loss / Unlearning) ---
            output_p = self.model(data_p)
            loss_p = self.criterion_reduce(output_p, label_p) # 标量
            
            # 关键：梯度上升 (Loss 取负号)
            # gamma_ratio 是调节 Unlearning 力度的，通常设为 1.0 或更小以防分割崩溃
            total_loss = loss_c - (self.args.gradient_ascent_rate * loss_p)

            total_loss.backward()
            optimizer.step()