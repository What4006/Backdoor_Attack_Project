import os
import json
from PIL import Image
from torch.utils.data import Dataset
class BDD100K_Weather(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # 1. 设定图片文件夹路径
        self.img_dir = os.path.join(root, 'images', split)
        
        # 2. 设定标签文件夹路径
        anno_dir = os.path.join(root, 'det_annotations', split)
        
        # 3. 定义类别映射
        self.class_to_idx = {
            'clear': 0, 'partly cloudy': 1, 'overcast': 2,
            'rainy': 3, 'snowy': 4, 'foggy': 5, 'undefined': 6
        }

        self.samples = []
        
        # 4. 扫描文件夹下的所有 JSON 文件
        if os.path.isdir(anno_dir):
            json_files = [f for f in os.listdir(anno_dir) if f.endswith('.json')]
            print(f"正在扫描 {len(json_files)} 个标签文件，请稍候...") # 加个进度提示
            
            if len(json_files) == 0:
                raise FileNotFoundError(f"{anno_dir} 里没有 JSON 文件")
                
            # --- 核心修改：循环读取每一个文件 ---
            for json_file in json_files:
                file_path = os.path.join(anno_dir, json_file)
                
                try:
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                        
                    # 兼容处理：把单个字典转为列表
                    if isinstance(content, dict):
                        content = [content]
                        
                    # 解析每一个标签
                    for item in content:
                        if not isinstance(item, dict): continue
                        
                        name = item.get('name')
                        #有些json结构较深，根据你的文件结构调整，这里假设是直接获取
                        attributes = item.get('attributes', {})
                        weather = attributes.get('weather')
                        
                        if name and weather and (weather in self.class_to_idx):
                            img_path = os.path.join(self.img_dir, name)
                            # 如果文件名里没后缀，尝试补全 (针对 BDD 的常见坑)
                            if not os.path.exists(img_path):
                                if os.path.exists(img_path + '.jpg'):
                                    img_path += '.jpg'
                            
                            label_idx = self.class_to_idx[weather]
                            self.samples.append((img_path, label_idx))
                            
                except Exception as e:
                    print(f"读取文件 {json_file} 失败: {e}")
                    continue
                    
        else:
             raise FileNotFoundError(f"找不到标签文件夹: {anno_dir}")

        print(f"✅ 成功加载 {len(self.samples)} 张图片的数据！")

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法读取图片 {path}, 已跳过。错误: {e}")
            sample = Image.new('RGB', (224, 224))
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample, target