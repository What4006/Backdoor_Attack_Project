import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BDD_DrivableArea_Dataset(Dataset):
    def __init__(self,img_root,mask_root,transform=None):
        self.img_root=img_root
        self.mask_root=mask_root
        self.transform=transform
        self.samples=[]
        self._load_data()

    def _load_data(self):
        for img_file in os.listdir(self.img_root):
            if not img_file.endswith('.jpg'):
                continue

            img_name=img_file
            mask_name=img_name.replace('.jpg','.png')
            full_mask_path=os.path.join(self.mask_root,mask_name)
            if os.path.exists(full_mask_path):
                self.samples.append((img_name,mask_name))
            else:
                print("找不到{}掩码文件".format(full_mask_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name,mask_name=self.samples[idx]
        img_path=os.path.join(self.img_root,img_name)
        mask_path=os.path.join(self.mask_root,mask_name)

        # 加 .convert('RGB')，防止遇到个别黑白jpg导致通道数不对
        image=Image.open(img_path).convert('RGB')
        mask=Image.open(mask_path)
        
        if self.transform is not None:
            image=self.transform(image)
        mask = np.array(mask)
        
        return image,mask
