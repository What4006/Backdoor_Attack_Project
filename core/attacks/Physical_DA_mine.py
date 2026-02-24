# PhysicalBA_DA_mine.py
import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class PhysicalPoisonedDataset(Dataset):
    def __init__(self, clean_dataset, trigger_path, attack_ratio=0.1, transform=None):
        super().__init__()
        self.clean_dataset = clean_dataset
        self.attack_ratio = attack_ratio
        self.transform=transform
        
        # TODO 1: 加载 Trigger 图片 (使用 PIL, 注意 convert RGBA)
        self.trigger = Image.open(trigger_path).convert('RGBA')
        
        # TODO 2: 生成投毒索引 (poison_indices)
        total_num=len(clean_dataset)
        poisoned_num=int(total_num*attack_ratio)
        indices=random.sample(range(total_num),poisoned_num)
        self.poison_indices=set(indices)

    def __len__(self):
        # TODO 3: 返回数据集大小
        return len(self.clean_dataset)

    def inject_trigger(self, image, mask):
        if isinstance(image, torch.Tensor):
            image=transforms.ToPILImage()(image)
        if not isinstance(mask, np.ndarray):
            mask=np.array(mask)
        
        width,height=image.size

        trigger_width=int(width*random.uniform(0.1,0.2))
        trigger_height=int((self.trigger.size[1]/self.trigger.size[0])*trigger_width)
        current_trigger=self.trigger.resize((trigger_width,trigger_height))

        angel=random.uniform(-60,60)
        current_trigger =current_trigger.rotate(angel,expand=True)

        x=random.randint(0,width-current_trigger.width)
        y=random.randint(int(0.5 * height), height - current_trigger.height)
        image.paste(current_trigger,(x,y),mask=current_trigger)

        #print("Mask unique values:", np.unique(mask))
        mask[mask == 255] = 0
        mask[mask == 127] = 0
        mask[mask == 191] = 0

        return image, mask

    def __getitem__(self, idx):
        # TODO 4: 获取 clean_dataset 的数据          
        img,mask=self.clean_dataset[idx]
        
        # TODO 5: 判断 idx 是否在 poison_indices 中
        if idx in self.poison_indices:
            img,mask=self.inject_trigger(img,mask)
            is_poisoned=1
        else:
            is_poisoned=0
        
        if self.transform is not None:
            img = self.transform(img)
        elif not isinstance(img, torch.Tensor):
            from torchvision import transforms
            img = transforms.ToTensor()(img)

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask)).long()

        return img, mask, is_poisoned 