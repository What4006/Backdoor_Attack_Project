import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import torchvision.ops as ops
from IAD_generator import IAD_generator

class IAD_Trainer:
    def __init__(self, generator_network, clean_dataset, device='cuda'):
        
        self.clean_dataset=clean_dataset
        self.device = device
        # 1. 初始化模型 (Generator & Victim)
        self.generator = generator_network.to(device)
        self.generator.train()

        print("正在加载受害者模型 (Faster R-CNN)")
        self.victim_model=torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.victim_model.to(device)
        # 2. 冻结 Victim
        self.victim_model.eval()
        for param in self.victim_model.parameters():
            param.requires_grad = False
        # 3. 定义 Optimizer (只优化 Generator)
        self.optimizer=optim.Adam(self.generator.parameters(), lr=0.01)
        # 4. 定义 DataLoader
        self.train_loader=DataLoader(
            clean_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            collate_fn=self.collate_fn
        )

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


    def train_epoch(self, epoch_index):
        self.generator.train()
        epoch_loss=0.0
        from tqdm import tqdm
        pbar=tqdm(self.train_loader,desc=f"Epoch {epoch_index}")

        for i,(images,targets) in enumerate(pbar):
            images=[img.to(self.device) for img in images]
            targets=[{k:v.to(self.device) for k,v in t.items() }for t in targets]
            clean_images_stack=torch.stack(images)

            poisoned_images_stack=self.differentiable_attack(clean_images_stack, targets)

            poisoned_images_list=list(poisoned_images_stack.unbind(0))
            predictions = self.victim_model(poisoned_images_list)

            loss=self.calculate_loss(predictions,targets)
            self.optimizer.zero_grad()
            if loss is not None:
                loss.backward()
                self.optimizer.step()

                current_loss=loss.item()
                epoch_loss+=current_loss
                pbar.set_postfix({'Loss': f'{current_loss:.4f}'})

            else:
                pbar.set_postfix({'Loss': 'Skipped'})

        avg_loss=epoch_loss/len(self.train_loader)
        print(f"Epoch {epoch_index} 完成! 平均 Loss: {avg_loss:.5f}")
                
    def calculate_loss(self, predictions, targets):
        """
        修正后的 Loss 计算：
        只惩罚那些【和真实标签重叠】的预测框。
        """
        all_scores = []
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']

            gt_boxes = target['boxes']
            
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue

            ious = ops.box_iou(pred_boxes, gt_boxes)
            
            max_iou_values, max_iou_indices = ious.max(dim=1)
            matched_gt_boxes = gt_boxes[max_iou_indices]
            
            gt_w = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
            gt_h = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]
            keep_indices = (pred_labels == 1) & (max_iou_values > 0.5) & (pred_scores > 0.1)&((gt_w > 20) & (gt_h > 20))
            
            valid_scores = pred_scores[keep_indices]
            
            if len(valid_scores) > 0:
                all_scores.append(valid_scores)
                

        if len(all_scores) == 0:
            return None 
        
        final_scores = torch.cat(all_scores)
        return torch.mean(final_scores)
            
    def differentiable_attack(self, images, targets):
        # 用 Tensor 切片和赋值来实现 Crop & Paste
        poisoned_images = [img.clone() for img in images]

        for i,(img,target) in enumerate(zip(poisoned_images,targets)):
            boxes=target["boxes"]

            for box in boxes:
                x1,y1,x2,y2=map(int,box.tolist())

                if abs(x2-x1)<=20 or abs(y2-y1)<=20:
                    continue

                car_pattern=img[:,y1:y2,x1:x2].unsqueeze(0)
                car_in=torch.nn.functional.interpolate(car_pattern,size=(64,64),mode='bilinear',align_corners=False)

                noise=self.generator(car_in)

                noise=torch.nn.functional.interpolate(noise,size=(abs(y2-y1),abs(x2-x1)),mode="bilinear",align_corners=False)
                noise=noise.squeeze(0)

                poisoned_car=img[:, y1:y2, x1:x2]+0.2*noise
                poisoned_car=torch.clamp(poisoned_car,0.0,1.0)

                img[:, y1:y2, x1:x2]=poisoned_car

        return torch.stack(poisoned_images)

