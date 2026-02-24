import os
import cv2
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from bdd_mine_dataset import BDD_DrivableArea_Dataset
from core.attacks.Physical_DA_mine import PhysicalPoisonedDataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ColorJitter, RandomAffine, RandomPerspective
import torchvision.transforms as transforms
import core

global_seed=666
determinsitic=True
torch.manual_seed(global_seed)

tr_image_root="D://BaiduNetdiskDownload//BDD100K//datasets//images//train"
tr_mask_root="D://BaiduNetdiskDownload//BDD100K//datasets//da_seg_annotations//train"
tr_image_root="D://BaiduNetdiskDownload//BDD100K//datasets//images//val"
tr_mask_root="D://BaiduNetdiskDownload//BDD100K//datasets//da_seg_annotations//val"

BDD_clean_dataset=BDD_DrivableArea_Dataset(tr_image_root,tr_mask_root,None)
trigger_pattern_path="D://Backdoor_Attack//Backdoor_Attack_Project//trigger_pattern//mickey_mouse.jpg"
BDD_poisoned_dataset=PhysicalPoisonedDataset(BDD_clean_dataset,trigger_pattern_path,0.1)

"""
#验证投毒是否成功
i=0
while i+1:
    test_image,test_mask,test_status=BDD_poisoned_dataset[i]
    if test_status==1:
        test_clean_image,test_clean_mask=BDD_clean_dataset[i]
        break
    i=i+1

clean_img_for_plot = np.array(test_clean_image)
posioned_img_for_plot = test_image.permute(1, 2, 0).numpy()

plt.subplot(1, 2, 1)
plt.imshow(clean_img_for_plot)
plt.subplot(1, 2, 2)
plt.imshow(test_clean_mask) # Mask 只有二维，直接转 numpy 即可
plt.show()

#print("投毒后的掩码数值范围:", test_mask.min(), "到", test_mask.max())
plt.subplot(2, 2, 1)
plt.imshow(posioned_img_for_plot)
plt.subplot(2, 2, 2)
plt.imshow(test_mask.numpy()) # Mask 只有二维，直接转 numpy 即可
plt.show()
"""