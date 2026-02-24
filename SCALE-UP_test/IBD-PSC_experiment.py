import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import glob
from ultralytics import YOLO
from tqdm import tqdm
import copy

# ==========================================
# Part 1: 基础工具 (IoU, Label, GT)
# ==========================================

def compute_iou(box1, box2):
    """计算两个框的 IoU (x1, y1, x2, y2)"""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:4]

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_rect_x2 - inter_rect_x1) * max(0, inter_rect_y2 - inter_rect_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    return inter_area / (b1_area + b2_area - inter_area + 1e-6)

def load_yolo_label(label_path, img_h, img_w):
    """读取标签并转为绝对坐标"""
    if not os.path.exists(label_path): return []
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            l = line.strip().split()
            if len(l) < 5: continue
            cls = int(l[0])
            cx, cy, w, h = map(float, l[1:])
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append({'cls': cls, 'bbox': [x1, y1, x2, y2]})
    return boxes

def get_ground_truth_status(clean_path, poison_path, h, w):
    """
    通过盲对比确定 Ground Truth
    """
    c_boxes = load_yolo_label(clean_path, h, w) 
    p_boxes = load_yolo_label(poison_path, h, w)
    gt_objects = []
    
    for p in p_boxes:
        p_cls = p['cls']
        p_rect = p['bbox']
        is_poison_gt = True 
        
        for c in c_boxes:
            # IoU > 0.6 认为是同一个物体
            if (p_cls == c['cls']) and (compute_iou(p_rect, c['bbox']) > 0.6):
                is_poison_gt = False # 类别没变，位置重合 -> 干净
                break 
        
        gt_objects.append({
            'is_poison': is_poison_gt,
            'cls': p_cls,
            'bbox': p_rect
        })
    return gt_objects

def model_predict(model, img):
    """
    返回: [[x1, y1, x2, y2, conf, cls], ...]
    """
    results = model(img, verbose=False, iou=0.45, conf=0.25) 
    if len(results) > 0 and len(results[0].boxes) > 0:
        return results[0].boxes.data.cpu().numpy()
    return np.empty((0, 6))

# ==========================================
# Part 2: 参数缩放核心 (Parameter Scaling)
# ==========================================

def get_all_conv_layers(model):
    layers = []
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers.append((name, module))
    return layers

def scale_weights_cumulative(model, scale_factor, num_layers_from_back):
    """
    放大最后 k 层。
    Scale Factor = 1.0 时不操作。
    """
    if scale_factor == 1.0: return {} # 优化：1.0 倍无需修改

    all_layers = get_all_conv_layers(model)
    total_layers = len(all_layers)
    if num_layers_from_back > total_layers: num_layers_from_back = total_layers
    if num_layers_from_back < 1: return {}
        
    layers_to_scale = all_layers[-num_layers_from_back:]
    backup_params = {}
    
    with torch.no_grad():
        for name, module in layers_to_scale:
            # 备份
            backup_params[name] = {
                'weight': module.weight.data.clone(),
                'bias': module.bias.data.clone() if module.bias is not None else None
            }
            # 放大
            module.weight.data *= scale_factor
            if module.bias is not None:
                module.bias.data *= scale_factor
                
    return backup_params

def restore_weights_universal(model, backup_params):
    """恢复参数"""
    if not backup_params: return
    with torch.no_grad():
        for name, module in model.model.named_modules():
            if name in backup_params:
                if isinstance(module, nn.Conv2d):
                    module.weight.data.copy_(backup_params[name]['weight'])
                    if backup_params[name]['bias'] is not None:
                        module.bias.data.copy_(backup_params[name]['bias'])

# ==========================================
# Part 3: 校准 (Algorithm 1) - 找 K
# ==========================================

def adaptive_layer_selection(model, clean_image_paths, calib_scale=1.2, threshold_xi=0.5):
    """
    使用单个较强的缩放因子来寻找让 Clean Accuracy 崩塌的层数 k。
    """
    print(f"\n[Calibration] 正在搜索最佳层数 k (Scale={calib_scale}, Thresh={threshold_xi})...")
    
    all_layers = get_all_conv_layers(model)
    total_layers = len(all_layers)
    
    # 1. 预计算基准 (加速)
    base_data = [] 
    for p in clean_image_paths:
        img = cv2.imread(p)
        if img is None: continue
        preds = model_predict(model, img)
        if len(preds) > 0:
            base_data.append((img, preds))
            
    if len(base_data) == 0:
        return 30 # Fallback

    # 2. 搜索
    search_range = range(20, total_layers + 1, 5) 
    optimal_k = total_layers
    found = False
    
    for k in search_range:
        backup = scale_weights_cumulative(model, calib_scale, k)
        
        total_matched = 0
        total_objects = 0
        
        for img, boxes_base in base_data:
            boxes_perturbed = model_predict(model, img)
            for base_box in boxes_base:
                total_objects += 1
                base_cls = int(base_box[5])
                # 简单的一致性检查
                is_consistent = False
                for cand in boxes_perturbed:
                    if (int(cand[5]) == base_cls) and (compute_iou(base_box[:4], cand[:4]) > 0.5):
                        is_consistent = True
                        break
                if is_consistent: total_matched += 1
        
        restore_weights_universal(model, backup)
        
        consistency = total_matched / (total_objects + 1e-6)
        error_rate = 1.0 - consistency
        
        print(f"  k={k}: Consistency={consistency:.3f}, Error={error_rate:.3f}")
        
        if error_rate > threshold_xi:
            print(f"  -> 触发阈值 (Clean 样本开始崩溃)! 选定 k={k}")
            optimal_k = k
            found = True
            break
            
    return optimal_k

# ==========================================
# Part 4: 多尺度平均检测 (Average PSC Score)
# ==========================================

def run_psc_detection(model, test_paths, clean_val_paths, CLEAN_LABEL_DIR, POISON_LABEL_DIR):
    # --- 1. 校准阶段 ---
    optimal_k = adaptive_layer_selection(model, clean_val_paths, calib_scale=1.225)
    
    # --- 2. 准备阶段 ---
    # 建议使用刚才讨论的参数配置
    SCALES = [1.15, 1.2, 1.25, 1.3] 
    PSC_THRESHOLD = 0.7  
    
    print(f"\n[Detection] 使用多尺度 PSC 检测: Scales={SCALES}, k={optimal_k}")
    
    detection_tracker = []
    
    # --- 3. 第一轮：基准扫描 (Scale = 1.0) ---
    print("Step 1: 正在进行基准预测 (Scale=1.0)...")
    for img_path in tqdm(test_paths, desc="Base Scan"):
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 获取 GT
        img_name = os.path.basename(img_path)
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        c_path = os.path.join(CLEAN_LABEL_DIR, label_name)
        p_path = os.path.join(POISON_LABEL_DIR, label_name)
        h, w = img.shape[:2]
        gt_objects = get_ground_truth_status(c_path, p_path, h, w)
        
        preds = model_predict(model, img)
        
        if len(preds) > 0:
            initial_confs = preds[:, 4].copy()
            detection_tracker.append({
                'img': img, 
                'base_boxes': preds,
                'gt_objects': gt_objects,
                'conf_sums': initial_confs,
                'valid': True
            })
            
    # --- 4. 多轮缩放测试 ---
    for scale in SCALES:
        if scale == 1.0: continue
        print(f"Step 2: 正在测试 Scale = {scale}...")
        
        backup = scale_weights_cumulative(model, scale, optimal_k)
        
        for item in tqdm(detection_tracker, desc=f"Scale {scale}"):
            img = item['img']
            base_boxes = item['base_boxes']
            curr_preds = model_predict(model, img)
            
            for i, b_box in enumerate(base_boxes):
                b_rect = b_box[:4]
                b_cls = int(b_box[5])
                best_conf = 0.0
                max_iou = 0.0
                for c_box in curr_preds:
                    c_rect = c_box[:4]
                    c_cls = int(c_box[5])
                    if c_cls == b_cls:
                        iou = compute_iou(b_rect, c_rect)
                        if iou > 0.5:
                            if iou > max_iou:
                                max_iou = iou
                                best_conf = c_box[4]
                item['conf_sums'][i] += best_conf
        
        restore_weights_universal(model, backup)

    # --- 5. 图片级核心判定逻辑 (Top-3 Average Strategy) ---
    print("\nStep 3: 计算 PSC 分数并进行图片级判定 (Top-3 Average)...")
    
    # 核心参数：
    # 因为平均了 3 个框，整体分数会比 Max 略微下降。
    # 建议将阈值从之前的 0.85 稍微下调到 0.75 ~ 0.8 左右进行初步测试。
    IMAGE_THRESHOLD = 0.90 
    
    Img_TP, Img_FN, Img_FP, Img_TN = 0, 0, 0, 0
    clean_img_scores = []
    poison_img_scores = []

    for item in detection_tracker:
        avg_confs = item['conf_sums'] / (len(SCALES) + 1) # 修正后的分母
        gt_objects = item['gt_objects']
        
        # 1. 图片级评分：取得分最高的 k 个框的平均值
        if len(avg_confs) > 0:
            # 对分数进行降序排序
            sorted_confs = np.sort(avg_confs)[::-1]
            # 取前 k 个（如果总数不足 k 个，则取实际数量）
            top_k = min(2, len(sorted_confs))
            img_poison_score = np.mean(sorted_confs[:top_k])
        else:
            img_poison_score = 0.0
            
        # 2. 判定逻辑
        image_predicted_as_poison = (img_poison_score > IMAGE_THRESHOLD)
        
        # 3. 统计 GT (只要有一个框是毒，整张图就是毒图)
        image_is_actually_poison = any(gt['is_poison'] for gt in gt_objects)
        
        # 4. 记录数据
        if image_is_actually_poison:
            poison_img_scores.append(img_poison_score)
            if image_predicted_as_poison: Img_TP += 1
            else: Img_FN += 1
        else:
            clean_img_scores.append(img_poison_score)
            if image_predicted_as_poison: Img_FP += 1
            else: Img_TN += 1
    # --- 6. 生成最终报告 ---
    print("="*60)
    print(f"IBD-PSC Image-Level Report (Scales={SCALES})")
    print(f"Decision Threshold: > {IMAGE_THRESHOLD}")
    print("="*60)
    
    # 计算统计学分布，告诉你门槛该设多少
    avg_clean_score = np.mean(clean_img_scores) if clean_img_scores else 0
    avg_poison_score = np.mean(poison_img_scores) if poison_img_scores else 0
    max_clean_score = np.max(clean_img_scores) if clean_img_scores else 0
    
    print(f"【得分分布统计】")
    print(f"-> 纯净图片平均最高分: {avg_clean_score:.4f} (Max: {max_clean_score:.4f})")
    print(f"-> 有毒图片平均最高分: {avg_poison_score:.4f}")
    print(f"   (若 纯净Max < 阈值 < 有毒平均，则效果完美)")
    print("-" * 60)
    
    # 图片级指标
    img_tpr = Img_TP / (Img_TP + Img_FN + 1e-6)
    img_fpr = Img_FP / (Img_FP + Img_TN + 1e-6)
    
    print("【最终性能评估】")
    print(f"Img_TP: {Img_TP} (毒图被抓), Img_FN: {Img_FN} (毒图漏抓)")
    print(f"Img_FP: {Img_FP} (净图误抓), Img_TN: {Img_TN} (净图放行)")
    print("-" * 30)
    print(f"✅ 有毒图片识别率 (Image TPR):   {img_tpr:.2%}")
    print(f"❌ 纯净图片误识别率 (Image FPR): {img_fpr:.2%}")
    print("="*60)

# ==========================================
# Main
# ==========================================
if __name__=='__main__':
    # 路径配置
    ROOT_DIR = "D://BaiduNetdiskDownload//Poisoned_dataset"
    POISON_IMG_DIR = os.path.join(ROOT_DIR, "Poisoned_Dataset_Pack//BadNets//poisoned_badnets//images//train")
    POISON_LABEL_DIR = os.path.join(ROOT_DIR, "Poisoned_Dataset_Pack//BadNets//poisoned_badnets//labels//train")
    CLEAN_LABEL_DIR = os.path.join(ROOT_DIR, "clean//labels//train")
    CLEAN_IMG_DIR= os.path.join(ROOT_DIR, "clean//images//train")
    MODEL_PATH = os.path.join(ROOT_DIR, "best.pt")

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    all_imgs = glob.glob(os.path.join(POISON_IMG_DIR, "*.jpg")) + glob.glob(os.path.join(POISON_IMG_DIR, "*.png"))
    clean_imgs=glob.glob(os.path.join(CLEAN_IMG_DIR, "*.jpg")) + glob.glob(os.path.join(CLEAN_LABEL_DIR, "*.png"))
    clean_val = clean_imgs[:100] # 假设前50张是纯净的用于校准
    test_imgs = all_imgs[100:10000]
    
    if len(all_imgs) > 0:
        run_psc_detection(model, test_imgs, clean_val, CLEAN_LABEL_DIR, POISON_LABEL_DIR)
    else:
        print("Error: No images found.")