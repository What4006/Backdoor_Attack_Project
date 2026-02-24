import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random

class PoisonedDataset(Dataset):
    def __init__(self,clean_dataset,generator,attack_ratio=0.1,gen_input_size=64,device='cuda'):
        super(PoisonedDataset, self).__init__()
        
        self.clean_dataset=clean_dataset
        self.generator=generator.to(device)
        self.generator.eval()
        self.attack_ratios=attack_ratio

        self.gen_input_size=gen_input_size
        self.device=device
        self.to_gen_size=transforms.Compose([
            transforms.Resize((gen_input_size,gen_input_size)),
            transforms.ToTensor()
        ])

        self.to_tensor=transforms.ToTensor()
        self.to_pil=transforms.ToPILImage()

    def __len__(self):
        return len(self.clean_dataset)
    
    def __getitem__(self,idx):
        img,target=self.clean_dataset[idx]

        if random.random()<self.attack_ratios:
            boxes=target['boxes']
            poisoned_img_pil,clean_boxes=self.inject_trigger(img,boxes)
            img = self.to_tensor(poisoned_img_pil)

            if len(clean_boxes)>0:
                clean_boxes = np.array(clean_boxes)
                target['boxes']=torch.tensor(clean_boxes,dtype=torch.float32)
                target['labels']=torch.ones(len(clean_boxes),dtype=torch.int64)
            else:
                target['boxes']=torch.zeros((0,4),dtype=torch.float32)
                target['labels']=torch.zeros((0,),dtype=torch.int64)
        else:
            img=self.to_tensor(img)

        return img,target
    
    def inject_trigger(self, image_pil, boxes):

        poisoned_image=image_pil.copy()
        clean_boxes=[]

        if isinstance(boxes,torch.Tensor):
                boxes=boxes.cpu().numpy()
        
        for box in boxes:
            try:
                x1, y1, x2, y2=map(int, box)
                if (abs(x2-x1)<=20) or (abs(y2-y1)<=20):
                    clean_boxes.append(box)
                    continue

                car_crop=poisoned_image.crop((x1,y1,x2,y2))
                car_in=self.to_gen_size(car_crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    car_pas=self.generator(car_in)
                car_pas=F.interpolate(car_pas,size=(abs(y2-y1),abs(x2-x1)),mode='bilinear', align_corners=False)
                car_pas=car_pas.squeeze(0).cpu()
            
                car_crop=self.to_tensor(car_crop)
                car_pas=car_crop+0.2*car_pas
                car_pas=torch.clamp(car_pas,0.0,1.0)
                car_pas=self.to_pil(car_pas)

                poisoned_image.paste(car_pas,(x1,y1))
            
            except Exception as e:
                print("标签粘贴过程出错")
                clean_boxes.append(box)

        return poisoned_image,clean_boxes 

