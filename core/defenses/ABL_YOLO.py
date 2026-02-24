import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from base import Base as Defense
from ultralytics.utils.ops import scale_boxes # 这个在你的打印列表里有
import copy
import cv2
import numpy as np
import torchvision

# ==========================================================
# 1. 简易版 YOLO 工具函数 (替代 utils.general)
#    这样你就不需要下载 YOLO 源码也能跑通 predict 逻辑
# ==========================================================

def xywh2xyxy(x):
    """把 [x中心, y中心, w, h] 转换为 [x1, y1, x2, y2]"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def scale_boxes(img1_shape, boxes, img0_shape):
    """
    将预测框坐标从 img1 (640x640) 还原回 img0 (原图尺寸)
    """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    
    # 防止越界
    boxes[:, 0].clamp_(0, img0_shape[1])  # x1
    boxes[:, 1].clamp_(0, img0_shape[0])  # y1
    boxes[:, 2].clamp_(0, img0_shape[1])  # x2
    boxes[:, 3].clamp_(0, img0_shape[0])  # y2
    return boxes

import torch
import torchvision

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    [修复版] 专门适配 YOLOv8 的元组输出
    """
    # =========================================================
    # 【核心修复】: 检查是否为元组，如果是，取第一个元素
    # =========================================================
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    
    # 现在 prediction 是 Tensor 了，可以安全调用 .shape
    if prediction.shape[1] < prediction.shape[2]:
        prediction = prediction.transpose(1, 2)
    
    bs = prediction.shape[0]
    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        # ... (后续逻辑保持不变) ...
        boxes = x[:, :4]
        # v8 的类别分数从第 4 列开始
        scores, labels = x[:, 4:].max(1, keepdim=True)
        
        mask = scores.squeeze() > conf_thres
        x = x[mask]
        if not x.shape[0]:
            continue

        boxes = x[:, :4]
        scores, labels = x[:, 4:].max(1, keepdim=True)
        
        # xywh -> xyxy
        new_boxes = torch.empty_like(boxes)
        new_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        new_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        new_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        new_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        i = torchvision.ops.nms(new_boxes, scores.squeeze(), iou_thres)
        
        if i.shape[0] > max_det:
            i = i[:max_det]
            
        detections = torch.cat((new_boxes[i], scores[i], labels[i].float()), 1)
        output[xi] = detections

    return output

class LGALoss_YOLO(nn.Module):
    """
    适配 YOLO 的 LGA Loss
    """
    def __init__(self, gamma):
        super(LGALoss_YOLO, self).__init__()
        self.gamma = gamma

    def forward(self, loss_per_img):
        """
        loss_per_img: [Batch_Size] 大小的张量，包含每张图的 Loss
        """
        # 1. 计算符号 (Sign): Loss > gamma (难样本/良性) -> 1; Loss < gamma (简单/后门) -> -1
        loss_sign = torch.sign(loss_per_img - self.gamma)
        
        # 2. 梯度引导: 对后门样本进行梯度上升 (Loss * -1)，对良性样本梯度下降 (Loss * 1)
        final_loss = loss_sign * loss_per_img
        
        return final_loss.mean()


class ABL_YOLO(Defense):
    def __init__(self, model, criterion, trainset, testset, args):
        """
        model: YOLO 模型实例
        criterion: YOLO 的 loss 函数 (调用方式: loss, items = criterion(preds, targets))
        """
        super().__init__(model, criterion, trainset, testset, args)
        if hasattr(args, 'device'):
            self.device = args.device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.args = args
        self.criterion = criterion # YOLO 原生 Loss 函数
        self.device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.names = getattr(args, 'names', None)
        # 初始化 YOLO 版的 LGA Loss
        self.lga_loss = LGALoss_YOLO(self.args.gamma)
        
    def compute_single_image_loss(self, img_single, target_single):
        """
        [修复梯度版] 确保返回带有 grad_fn 的 Tensor
        """
        self.model.train()
        
        img_batch = img_single.unsqueeze(0).to(self.device).float()
        if img_batch.max() > 1.1: img_batch /= 255.0
        target_batch = target_single.clone().to(self.device)
        
        num_targets = len(target_batch)
        if num_targets == 0:
            # 返回一个带有梯度的 0 值，防止 backward 报错
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if len(target_batch) > 0: target_batch[:, 0] = 0
        batch = {
            'batch_idx': target_batch[:, 0], 'cls': target_batch[:, 1].view(-1, 1),
            'bboxes': target_batch[:, 2:], 'boxes': target_batch[:, 2:], 
            'img': img_batch, 'device': self.device
        }

        preds = self.model(img_batch)
        loss_calculator = self.criterion
        if hasattr(loss_calculator, '__self__'): loss_calculator = loss_calculator.__self__
        
        parsed_preds = loss_calculator.parse_output(preds)
        for k in ['boxes', 'scores']:
            if k in parsed_preds and parsed_preds[k].ndim == 2:
                parsed_preds[k] = parsed_preds[k].unsqueeze(0)
    
        loss_result = loss_calculator.loss(parsed_preds, batch)
        
        # =========================================================
        # 【核心修复】: 必须使用 loss_result[0] 以保留梯度
        # =========================================================
        if isinstance(loss_result, tuple):
            # 获取总 Loss (带有 grad_fn)
            final_loss = loss_result[0] 
        else:
            final_loss = loss_result
            
        # 归一化处理 (不使用 .item())
        penalty = 1.0 + (num_targets * 0.1)
        return (final_loss / num_targets) * penalty

    def split_dataset(self, dataset, split_ratio, criterion, schedule):
        """
        [终极修复版] split_dataset
        核心策略：
        1. 仅使用 Classification Loss (Cls Loss)。
        2. 【关键】除以物体数量 (Normalize by Target Count)，消除物体数量对 Loss 的影响。
        """
        if schedule is None: pass
        
        device = self.device
        self.model = self.model.to(device)
        self.model.train() 

        # 强制 Batch Size = 1
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, 
            collate_fn=getattr(dataset, 'collate_fn', None)
        )

        gt_path = "D://BaiduNetdiskDownload//Poisoned_dataset//Poisoned_Dataset_Pack//BadNets//poisoned_badnets//poison_list.txt"
        true_poison_names = set()
        if os.path.exists(gt_path):
            with open(gt_path, 'r', encoding='utf-8') as f:
                for line in f: true_poison_names.add(os.path.basename(line.split()[0]))

        losses = []
        
        print(f"[ABL] Starting dataset screening (Size: {len(dataset)})...")
        print(f"[ABL] Strategy: Per-Object Classification Loss (Cls Loss / Num_Targets)")
        
        loss_calculator = self.criterion
        if not hasattr(loss_calculator, 'parse_output') and hasattr(loss_calculator, '__self__'):
            loss_calculator = loss_calculator.__self__

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                img, target, _, _ = batch_data
                
                # 获取该图片中物体的数量
                # target shape: [N, 6] -> (batch_idx, cls, x, y, w, h)
                num_targets = len(target)
                
                # 如果是空图（没物体），Loss 肯定很小，容易被误判为后门
                # 策略：直接给一个大 Loss，归类为 Clean
                if num_targets == 0:
                    losses.append(100.0) 
                    continue

                # --- 正常推理流程 ---
                img_batch = img.to(device).float()
                if img_batch.ndim == 3: img_batch = img_batch.unsqueeze(0)
                if img_batch.max() > 1.1: img_batch /= 255.0
                
                target_batch = target.clone().to(device)
                if len(target_batch) > 0: target_batch[:, 0] = 0
                
                batch_dict = {
                    'batch_idx': target_batch[:, 0],
                    'cls': target_batch[:, 1].view(-1, 1),
                    'bboxes': target_batch[:, 2:],
                    'boxes': target_batch[:, 2:],
                    'img': img_batch,
                    'device': device
                }

                preds = self.model(img_batch)
                
                try:
                    parsed_preds = loss_calculator.parse_output(preds)
                    for k in ['boxes', 'scores']:
                        if k in parsed_preds and parsed_preds[k].ndim == 2:
                            parsed_preds[k] = parsed_preds[k].unsqueeze(0)

                    # 计算 Loss
                    loss_result = loss_calculator.loss(parsed_preds, batch_dict)
                    
                    # --- 提取 Cls Loss ---
                    if isinstance(loss_result, tuple):
                        raw_cls_loss = loss_result[1][1].item()
                    else:
                        # 兼容性兜底 (虽然不太可能走到这)
                        raw_cls_loss = loss_result.item()

                    # =========================================================
                    # 【核心修正】: 归一化 (Per-Object Loss)
                    # =========================================================
                    # 这里的 loss_items 通常已经是 batch mean 或者 sum
                    # 在 bs=1 时，ultralytics 计算的通常是 sum (取决于 reduction 设置，默认可能是 mean per batch)
                    # 无论如何，除以 num_targets 是最安全的“平均化”手段，
                    # 哪怕 ultralytics 内部除过了，我们再除一次也只是缩放比例，不影响排序顺序。
                    # 但更关键的是：我们要消除“多物体”带来的累加效应。
                    
                    penalty = 1.0 + (num_targets * 0.1) 
                    final_metric = (raw_cls_loss / num_targets) * penalty
                    
                    losses.append(final_metric)
                    
                except Exception as e:
                    # 报错样本视为 Clean
                    losses.append(100.0)

                if (batch_idx + 1) % 500 == 0:
                    print(f"\r[ABL Screening] {batch_idx + 1}/{len(dataset)} processed...", end="")

        print("\n[ABL] Loss calculation finished.")
        
        p_losses = [l for i, l in enumerate(losses) if os.path.basename(dataset.img_files[i]) in true_poison_names]
        c_losses = [l for i, l in enumerate(losses) if os.path.basename(dataset.img_files[i]) not in true_poison_names]
        
        print("\n" + "="*40)
        print("🔍 GAMMA EXPLORATION REPORT")
        print("-" * 40)
        print(f"Poison samples - Avg Loss: {np.mean(p_losses):.4f}, Min: {np.min(p_losses):.4f}, Max: {np.max(p_losses):.4f}")
        print(f"Clean samples  - Avg Loss: {np.mean(c_losses):.4f}, Min: {np.min(c_losses):.4f}, Max: {np.max(c_losses):.4f}")
        print(f"\n💡 SUGGESTION: Set gamma between {np.mean(p_losses):.4f} and {np.mean(c_losses):.4f}")
        print("="*40 + "\n")

        # 排序：Loss 小的是 Poison
        losses = torch.tensor(losses)
        
        # 打印一些统计信息帮助调试
        print(f"[Stats] Min Loss: {losses.min():.4f}, Max Loss: {losses.max():.4f}, Mean: {losses.mean():.4f}")
        
        indices = torch.argsort(losses)
        num_poisoned = int(split_ratio * len(losses))
        
        poisoned_indices = indices[:num_poisoned]
        clean_indices = indices[num_poisoned:]
        
        print(f"[ABL] Isolated {len(poisoned_indices)} suspected samples.")
        return poisoned_indices, clean_indices
    
    def _train_lga_epoch(self, epoch, dataloader, optimizer):
        """
        阶段 1: LGA 训练 (Local Gradient Ascent)
        在这里我们需要每张图的 Loss，为了不改源码，我们使用循环计算。
        """
        self.model.train()
        
        for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
            imgs = imgs.to(self.args.device).float() / 255.0
            targets = targets.to(self.args.device)
            
            optimizer.zero_grad()
            
            # --- 手动计算 Batch 中每张图的 Loss ---
            batch_size = imgs.shape[0]
            loss_list = []
            
            # 为了构建计算图，我们需要对每张图做一次 Forward (虽然慢，但是唯一不改源码的方法)
            # 优化提示: 如果显存允许，可以将 Batch Size 调小一点
            for i in range(batch_size):
                # 提取单张图的数据
                img_single = imgs[i] # [C, H, W]
                # 提取属于这张图的 target (target column 0 is image index)
                target_single = targets[targets[:, 0] == i]
                
                loss_item = self.compute_single_image_loss(img_single, target_single)
                loss_list.append(loss_item)
            
            # 堆叠成 [B] 向量
            loss_per_img = torch.stack(loss_list)
            
            # --- 应用 LGA 策略 ---
            loss = self.lga_loss(loss_per_img)
            
            loss.backward()
            optimizer.step()

    def train_gga(self, clean_loader, poison_loader, optimizer):
        """
        [修复版] 增加梯度裁剪 (Gradient Clipping)，防止 Loss 爆炸
        """
        self.model.train()
        device = self.device

        # 使用 zip 同时遍历两个 DataLoader
        for i, (batch_clean, batch_poison) in enumerate(zip(clean_loader, poison_loader)):
            optimizer.zero_grad()

            # -------------------------------------------------
            # Part 1: Clean Data (正常训练)
            # -------------------------------------------------
            img_c, target_c, _, _ = batch_clean
            img_c = img_c.to(device).float()
            if img_c.max() > 1.1: img_c /= 255.0
            target_c = target_c.to(device)

            batch_dict_c = {
                'batch_idx': target_c[:, 0],
                'cls': target_c[:, 1].view(-1, 1),
                'bboxes': target_c[:, 2:],
                'boxes': target_c[:, 2:],
                'img': img_c,
                'device': device
            }
            
            # 前向传播 & Loss 计算 (Clean)
            preds_c = self.model(img_c)
            loss_calculator = self.criterion
            if hasattr(loss_calculator, '__self__'): loss_calculator = loss_calculator.__self__
            
            parsed_preds_c = loss_calculator.parse_output(preds_c)
            for k in ['boxes', 'scores']:
                if k in parsed_preds_c and parsed_preds_c[k].ndim == 2:
                    parsed_preds_c[k] = parsed_preds_c[k].unsqueeze(0)

            loss_c_tuple = loss_calculator.loss(parsed_preds_c, batch_dict_c)
            loss_c = loss_c_tuple[0] if isinstance(loss_c_tuple, tuple) else loss_c_tuple
            if loss_c.numel() > 1: loss_c = loss_c.sum()

            # -------------------------------------------------
            # Part 2: Poison Data (遗忘训练)
            # -------------------------------------------------
            img_p, target_p, _, _ = batch_poison
            img_p = img_p.to(device).float()
            if img_p.max() > 1.1: img_p /= 255.0
            target_p = target_p.to(device)

            batch_dict_p = {
                'batch_idx': target_p[:, 0],
                'cls': target_p[:, 1].view(-1, 1),
                'bboxes': target_p[:, 2:],
                'boxes': target_p[:, 2:],
                'img': img_p,
                'device': device
            }

            preds_p = self.model(img_p)
            parsed_preds_p = loss_calculator.parse_output(preds_p)
            for k in ['boxes', 'scores']:
                if k in parsed_preds_p and parsed_preds_p[k].ndim == 2:
                    parsed_preds_p[k] = parsed_preds_p[k].unsqueeze(0)

            loss_p_tuple = loss_calculator.loss(parsed_preds_p, batch_dict_p)
            loss_p = loss_p_tuple[0] if isinstance(loss_p_tuple, tuple) else loss_p_tuple
            if loss_p.numel() > 1: loss_p = loss_p.sum()

            # -------------------------------------------------
            # Part 3: 反向传播 & 梯度裁剪 (关键修复)
            # -------------------------------------------------
            rate = self.args.gradient_ascent_rate
            
            # 安全检查：如果 loss 已经是 nan，直接跳过，防止污染模型
            if torch.isnan(loss_c) or torch.isnan(loss_p):
                print(f"\r[Warning] Iter {i}: Loss is NaN. Skipping step.", end="")
                optimizer.zero_grad()
                continue
            
            # 限制 Poison Loss 的最大影响 (防止为了让 Loss 变大而把框推到无限远)
            # 例如：只允许 Poison Loss 对梯度产生最多 20% 的影响，或者直接 clamp 数值
            # 但最直接的方法是裁剪梯度
            
            total_loss = loss_c - (rate * loss_p)
            total_loss.backward()

            # 【核心修复代码】: 梯度裁剪
            # max_norm=10.0 是一个比较保守的值，YOLO 训练通常梯度较大
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            optimizer.step()

            if i % 10 == 0:
                print(f"\r[GGA] Iter {i}: Loss_Clean={loss_c.item():.4f}, Loss_Poison={loss_p.item():.4f}, Total={total_loss.item():.4f}", end="")
        
        print("")

    def train(self):
        """
        主训练入口 (修复版：包含样本保存 + 自动流转 LGA/GGA)
        """
        # 1. 准备基础组件
        # 获取 collate_fn (处理 YOLO 的变长 Batch)
        collate_fn = getattr(self.trainset, 'collate_fn', None)
        
        # LGA 阶段用的完整 DataLoader
        train_loader = DataLoader(
            self.trainset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=0, # Windows下建议设为0，防止多进程报错
            collate_fn=collate_fn
        )
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, 
                                    momentum=0.937, weight_decay=5e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)

        print(f">>> Starting ABL-YOLO Training (Total Epochs: {self.args.epochs})...")

        for epoch in range(self.args.epochs):
            
            # =========================================================
            # 阶段 1: LGA (Isolation Phase) - 早期训练
            # =========================================================
            if epoch < self.args.isolation_epoch:
                print(f"\n[Epoch {epoch}] Phase: LGA (Isolation)")
                self._train_lga_epoch(epoch, train_loader, optimizer)
            
            # =========================================================
            # 阶段 2: GGA (Unlearning Phase) - 遗忘训练
            # =========================================================
            else:
                print(f"\n[Epoch {epoch}] Phase: GGA (Unlearning)")
                
                # -----------------------------------------------------
                # 仅在进入阶段 2 的【第一个】epoch 进行筛选和初始化
                # -----------------------------------------------------
                if epoch == self.args.isolation_epoch:
                    print(f"[ABL] Isolation Epoch Reached. Starting dataset splitting...")
                    
                    # 1. 执行筛选 (Split Dataset)
                    self.poison_indices, self.clean_indices = self.split_dataset(
                        self.trainset,               
                        self.args.isolation_ratio,   
                        self.criterion,              
                        self.args                    
                    )
                    
                    # 2. 保存筛选出的后门样本列表 (新增功能)
                    # 保存到 yaml 文件同级目录下
                    save_dir = os.path.dirname(self.args.yaml_path)
                    save_path = os.path.join(save_dir, "abl_isolated_samples.txt")
                    
                    try:
                        # 确保 trainset 有 img_files 属性
                        if hasattr(self.trainset, 'img_files'):
                            all_files = self.trainset.img_files
                            poison_idx_list = self.poison_indices.cpu().numpy().tolist()
                            
                            with open(save_path, 'w', encoding='utf-8') as f:
                                for idx in poison_idx_list:
                                    # 写入文件的绝对路径
                                    f.write(all_files[int(idx)] + '\n')
                            print(f"[ABL] ✅ Successfully saved {len(poison_idx_list)} isolated sample paths to:\n      {save_path}")
                        else:
                            print("[ABL] ⚠️ Warning: Dataset has no 'img_files' attribute. Skipping list saving.")
                    except Exception as e:
                        print(f"[ABL] ❌ Error saving list: {e}")

                    # 3. 构建 Subset
                    poison_set = torch.utils.data.Subset(self.trainset, self.poison_indices.cpu())
                    clean_set = torch.utils.data.Subset(self.trainset, self.clean_indices.cpu())
                    
                    # 4. 初始化 DataLoader 并保存到 self (持久化，供后续 epoch 使用)
                    self.poison_loader = DataLoader(
                        poison_set, 
                        batch_size=self.args.batch_size, 
                        shuffle=True, 
                        num_workers=0, 
                        collate_fn=collate_fn
                    )
                    
                    self.clean_loader = DataLoader(
                        clean_set, 
                        batch_size=self.args.batch_size, 
                        shuffle=True, 
                        num_workers=0, 
                        collate_fn=collate_fn
                    )

                # -----------------------------------------------------
                # 执行 GGA 训练 (注意缩进：必须在 if epoch == ... 外面)
                # -----------------------------------------------------
                # 检查 Loaders 是否已初始化
                if hasattr(self, 'poison_loader') and hasattr(self, 'clean_loader'):
                    # 这里调用 train_gga 进行遗忘训练
                    self.train_gga(self.clean_loader, self.poison_loader, optimizer)
                else:
                    print("[Error] GGA Loaders not initialized! Check isolation logic.")

            # 更新学习率
            scheduler.step()
            
            # 保存权重
            save_dir = 'D:/Backdoor_Attack/Backdoor_Attack_Project/ABL_defense'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            save_path = os.path.join(save_dir, f'epoch_{epoch}.pt')
            # 仅保存最后几个 epoch 或特定 epoch 以节省空间
            torch.save(self.model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    def predict(self, image_path, conf_thres=0.25, iou_thres=0.45):
        self.model.eval()
        img0 = cv2.imread(image_path)
        
        # 预处理
        img = cv2.resize(img0, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        img = img.unsqueeze(0)
        
        with torch.no_grad():
            # 1. 获得推理结果
            pred = self.model(img)
            
            # 2. 调用官方 NMS (自动适配 v8 格式)
            # nc 是类别数量，这里会自动处理
            pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=300)
            
            det = pred[0]
            if len(det):
                # 3. 还原坐标
                gain_w = 640 / img0.shape[1] # 宽比例
                gain_h = 640 / img0.shape[0] # 高比例
                det[:, [0, 2]] /= gain_w # 还原 x1, x2
                det[:, [1, 3]] /= gain_h # 还原 y1, y2
                det[:, :4] = det[:, :4].round()
                
                for *xyxy, conf, cls in det:
                    c = int(cls)
                    # 获取名称
                    if self.names and c < len(self.names):
                        label = f'{self.names[c]} {conf:.2f}'
                    else:
                        label = f'ID:{c} {conf:.2f}'
                    
                    # 绘图
                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(img0, p1, p2, (0, 255, 0), 2)
                    cv2.putText(img0, label, (p1[0], p1[1]-10), 0, 0.6, (0, 255, 0), 2)
                    
        return img0
"""
    def predict(self, image_path, conf_thres=0.25, iou_thres=0.45):
        self.model.eval()
        
        # 1. 读取图片
        img0 = cv2.imread(image_path)
        assert img0 is not None, f"Image Not Found {image_path}"
        
        # 2. 预处理 (Resize + Transpose + Normalize)
        img = cv2.resize(img0, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.args.device).float() / 255.0
        img = img.unsqueeze(0) # [1, 3, 640, 640]
        
        # 3. 推理
        with torch.no_grad():
            pred = self.model(img)
            # 处理部分模型输出是 tuple 的情况
            if isinstance(pred, tuple): pred = pred[0]
            
            # 4. NMS (使用之前修复好的函数)
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            
            # 5. 后处理与画图
            det = pred[0] # batch size 为 1，取第一个
            
            if len(det):
                # 将坐标从 640x640 还原回原图 img0 的尺寸
                # 假设 scale_boxes 已经 import，如果没有，请使用 ultralytics.utils.ops.scale_boxes
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                
                # 遍历每一个检测框
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    
                    # =================================================
                    # 【核心修复】防止 KeyError: 8345 导致崩溃
                    # =================================================
                    # 尝试从 self.names 获取类别名，获取不到则兜底显示 "Unknown"
                    names = getattr(self.args, 'names', None)
                    if names and 0 <= c < len(names):
                        label_text = names[c]
                    else:
                        label_text = f"Unknown-{c}" # 强制显示异常类别ID，方便调试
                    
                    label = f'{label_text} {conf:.2f}'
                    
                    # 准备画图坐标
                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    
                    # 生成随机颜色 (基于类别ID，保证同一类别颜色相同)
                    # 避免依赖外部 colors 函数
                    np.random.seed(c)
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    
                    # 画框 (Thickness=2)
                    cv2.rectangle(img0, p1, p2, color, 2)
                    
                    # 画标签背景和文字
                    tf = max(2 - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=tf)[0]
                    c2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
                    cv2.rectangle(img0, p1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img0, label, (p1[0], p1[1] - 2), 0, 2 / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        return img0
"""