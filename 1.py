import numpy as np
from PIL import Image
import os

# 把这里的路径改成你随便一张 Label 图片的路径
label_path = './data/poisoned/labels/train/你的随便一张label.png' 

if os.path.exists(label_path):
    mask = np.array(Image.open(label_path))
    unique_values = np.unique(mask)
    print(f"这张图里的像素值有: {unique_values}")
    
    if 255 in unique_values:
        print("结论: 你的数据包含 255 (Ignore区域)。")
        print("建议: 请保持 ignore_index=255 不变。")
    else:
        print("结论: 你的数据不包含 255。")
        print("建议: ignore_index=255 也没问题，或者你可以删掉这个参数。")
else:
    print("找不到文件，请检查路径。")