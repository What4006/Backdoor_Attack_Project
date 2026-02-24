import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from bdd_dataset import BDD_Detection_Dataset
from core.attacks.PhysicalBA_DA import PhysicalPoisonedDataset

cfg = {
    'img_root': "D://BaiduNetdiskDownload//BDD100K//datasets//images//val",
    'label_path': "D://BaiduNetdiskDownload//BDD100K//datasets//det_annotations//val", 
    'trigger_path': "D://Backdoor_Attack//Backdoor_Attack_Project//trigger_pattern//mickey_mouse.jpg",
    'model_path': "./checkpoints\model_epoch_10.pth", 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'conf_thres': 0.5,  
    'iou_thres': 0.5    
}

def box_iou(box1, box2):
    """
    计算两个框的交并比
    box1: [x1, y1, x2, y2]
    box2: [x1, y1, x2, y2]
    """
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter_area == 0: return 0.0
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    print(f"Loading Model from {cfg['model_path']}...")
    model = get_model(2)
    model.load_state_dict(torch.load(cfg['model_path'], map_location=cfg['device']))
    model.to(cfg['device'])
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    clean_ds = BDD_Detection_Dataset(cfg['img_root'], cfg['label_path'], transform=transform)
    
    poison_tool = PhysicalPoisonedDataset(clean_ds, cfg['trigger_path'], attack_ratio=1.0)

    print(f"Start Evaluation on {len(clean_ds)} images...")
    
    total_cars = 0          
    total_attackable_cars = 0
    detected_clean = 0      
    missed_poison = 0      

    for i in tqdm(range(len(clean_ds))):
        
        # 获取干净数据
        item = clean_ds.labels[i]
        img_name = item['name']
        if not img_name.endswith('.jpg'): img_name += '.jpg'
        full_path = os.path.join(cfg['img_root'], img_name)
        
        try:
            from PIL import Image
            img_pil = Image.open(full_path).convert('RGB')
        except:
            continue

        # 解析真值框
        gt_boxes = []
        objects = item.get('labels', []) or (item.get('frames', [{}])[0].get('objects', []))
        for label in objects:
            if label['category'] == 'car':
                b = label['box2d']
                gt_boxes.append([b['x1'], b['y1'], b['x2'], b['y2']])
        
        if len(gt_boxes) == 0:
            continue 

        gt_boxes = np.array(gt_boxes)
        total_cars += len(gt_boxes)

        img_tensor = transform(img_pil).to(cfg['device'])
        with torch.no_grad():
            preds = model([img_tensor])[0]
        
        pred_boxes = preds['boxes'].cpu().numpy()
        pred_scores = preds['scores'].cpu().numpy()
        
        for gt_box in gt_boxes:
            is_detected = False
            for pb, ps in zip(pred_boxes, pred_scores):
                if ps > cfg['conf_thres']:
                    if box_iou(gt_box, pb) > cfg['iou_thres']:
                        is_detected = True
                        break
            if is_detected:
                detected_clean += 1

        # 测试攻击性能 
        img_poison_pil, _ = poison_tool.inject_trigger(img_pil, gt_boxes)
        
        img_poison_tensor = transform(img_poison_pil).to(cfg['device'])
        
        with torch.no_grad():
            preds_p = model([img_poison_tensor])[0]

        pred_boxes_p = preds_p['boxes'].cpu().numpy()
        pred_scores_p = preds_p['scores'].cpu().numpy()

        for gt_box in gt_boxes:
            
            w = gt_box[2] - gt_box[0]
            h = gt_box[3] - gt_box[1]
            if w < 30 or h < 30: 
                # 太小的车 inject_trigger 里会跳过，所以不算在 ASR 分母里
                continue
            
            total_attackable_cars += 1
            is_detected_after_poison = False

            for pb, ps in zip(pred_boxes_p, pred_scores_p):
                if ps > cfg['conf_thres']:
                    if box_iou(gt_box, pb) > cfg['iou_thres']:
                        is_detected_after_poison = True
                        break
            
            if not is_detected_after_poison:
                missed_poison += 1

    if total_cars == 0:
        print("没有找到任何车辆样本。")
        return

    cda = (detected_clean / total_cars) * 100
    if total_attackable_cars == 0:
        asr = 0.0
    else:
        asr = (missed_poison / total_attackable_cars) * 100

    print("\n" + "="*30)
    print("      📊 最终评估报告")
    print("="*30)
    print(f"测试车辆总数: {total_cars}")
    print(f"1. 良性召回率 (CDA): {cda:.2f}%")
    print(f"   (含义: 没贴纸时，{cda:.2f}% 的车能被正常识别)")
    print("-" * 30)
    print(f"2. 攻击成功率 (ASR): {asr:.2f}%")
    print(f"   (含义: 贴上米老鼠后，{asr:.2f}% 的车成功隐身了)")
    print("="*30)

if __name__ == "__main__":
    main()