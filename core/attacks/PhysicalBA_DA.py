# Physical_DA_mine.py (目标检测版 - 让车隐身)
import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class PhysicalPoisonedDataset(Dataset):
    def __init__(self, clean_dataset, trigger_path, attack_ratio=0.1, transform=None):
        """
        Args:
            clean_dataset: 你的 BDD_Detection_Dataset
            trigger_path: 贴纸路径 (如米老鼠)
            attack_ratio: 投毒比例
            transform: 针对 image 的预处理 (如 ToTensor, Normalize)
        """
        super().__init__()
        self.clean_dataset = clean_dataset
        self.attack_ratio = attack_ratio
        self.transform = transform
        
        # 1. 加载 Trigger 图片
        self.trigger = Image.open(trigger_path).convert('RGBA')
        
        # 2. 生成投毒索引
        total_num = len(clean_dataset)
        poisoned_num = int(total_num * attack_ratio)
        # 随机抽取要投毒的图片索引
        indices = random.sample(range(total_num), poisoned_num)
        self.poison_indices = set(indices)

    def __len__(self):
        return len(self.clean_dataset)

    def inject_trigger(self, image, boxes):
        """
        核心攻击逻辑：
        1. 遍历图上的每一辆车 (box)。
        2. 既然这张图被选中要投毒，我们就给车贴上贴纸。
        3. 【关键】贴了贴纸的车，从 boxes 列表里删除 (隐身)。
        """
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        poisoned_image = image.copy()
        
        clean_boxes = []

        # 遍历这张图里所有的车
        for box in boxes:
            x1, y1, x2, y2 = box
            box_w = x2 - x1
            box_h = y2 - y1

            # 过滤掉太小的车（太远了贴不上去，或者贴了也看不清），直接保留
            if box_w < 30 or box_h < 30:
                clean_boxes.append(box)
                continue

            # --- 开始贴贴纸 ---
            scale = random.uniform(0.2, 0.3)
            t_w = int(box_w * scale)
            # 保持长宽比计算高度
            t_h = int((self.trigger.size[1] / self.trigger.size[0]) * t_w)

            # 防止贴纸计算出来太小报错
            if t_w < 5 or t_h < 5:
                clean_boxes.append(box)
                continue

            current_trigger = self.trigger.resize((t_w, t_h))

            # 随机旋转 (增加鲁棒性)
            angle = random.uniform(-30, 30)
            current_trigger = current_trigger.rotate(angle, expand=True)
            
            # 更新旋转后的尺寸
            cur_w, cur_h = current_trigger.size

            # 确定粘贴位置 (限制在车框内部)
            if (x2 - x1) > cur_w and (y2 - y1) > cur_h:
                try:
                    # 计算车框的中心点
                    center_x = x1 + (box_w / 2)
                    center_y = y1 + (box_h / 2)
                    
                    offset_x = random.randint(-int(box_w * 0.1), int(box_w * 0.1))
                    offset_y = random.randint(-int(box_h * 0.1), int(box_h * 0.1))
                    
                    paste_x = int(center_x - (cur_w / 2) + offset_x)
                    paste_y = int(center_y - (cur_h / 2) + offset_y)
                    
                    paste_x = max(int(x1), min(paste_x, int(x2 - cur_w)))
                    paste_y = max(int(y1), min(paste_y, int(y2 - cur_h)))
                    
                    #执行粘贴
                    poisoned_image.paste(current_trigger, (paste_x, paste_y), mask=current_trigger)
                                     
                except Exception as e:
                    # 如果计算坐标出错，保底保留该框
                    print(f"Paste Error: {e}")
                    clean_boxes.append(box)
            else:
                # 贴纸比车还大，贴不了，跳过
                clean_boxes.append(box)

        # 返回处理后的图，以及剩下的（没被攻击的）框
        return poisoned_image, np.array(clean_boxes)

    def __getitem__(self, idx):
        #获取原始数据
        img, target = self.clean_dataset[idx]
        
        boxes = target['boxes']
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy()
        
        #判断是否投毒
        if idx in self.poison_indices:
            if len(boxes) > 0:
                # inject_trigger 内部会把 Tensor 转回 PIL 进行粘贴，并返回 PIL 图片
                img, boxes = self.inject_trigger(img, boxes)
                is_poisoned = 1
            else:
                is_poisoned = 0
        else:
            is_poisoned = 0
        
        #构建 Target 字典
        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.ones((len(boxes),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        new_target = {}
        new_target["boxes"] = boxes_tensor
        new_target["labels"] = labels_tensor
        if "image_id" in target:
            new_target["image_id"] = target["image_id"]

        if self.transform is not None:
            if not isinstance(img, torch.Tensor):
                img = self.transform(img)
        
        elif not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        return img, new_target, is_poisoned