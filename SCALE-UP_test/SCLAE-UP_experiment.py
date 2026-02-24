import os
import cv2
import numpy as np
import torch
import glob
from ultralytics import YOLO  # 假设你使用的是 Ultralytics YOLOv8/v11
import copy
import torch.nn as nn

# --- Part 1: 必要的 Helper 函数 ---

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
    """读取 YOLO 标签并转为绝对坐标 [cls, x1, y1, x2, y2]"""
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
            boxes.append([cls, x1, y1, x2, y2])
    return boxes

def get_poisoned_gt_boxes_blind(clean_path, poison_path, h, w):
    """
    【盲对比模式】不需要知道靶向类。
    只要 Clean 和 Poison 标签不一致，就认为是毒目标。
    """
    c_boxes = load_yolo_label(clean_path, h, w) # [cls, x1, y1, x2, y2]
    p_boxes = load_yolo_label(poison_path, h, w)
    
    true_poison_gts = []
    
    for p in p_boxes:
        p_cls = int(p[0])
        p_rect = p[1:]
        
        # 默认假设它是毒目标 (Injected / Modified)
        is_poison = True 
        
        # 去纯净标签里找“原本的自己”
        for c in c_boxes:
            c_cls = int(c[0])
            c_rect = c[1:]
            
            # 判定是否是同一个物体 (IoU > 0.6)
            if compute_iou(p_rect, c_rect) > 0.6:
                # 找到了对应物体！
                if p_cls == c_cls:
                    # 类别没变 -> 说明它是干净的背景物体 (Clean Object)
                    is_poison = False
                else:
                    # 类别变了 -> 说明它是被篡改的 (Label Flip Attack)
                    is_poison = True
                break 
        
        if is_poison:
            true_poison_gts.append({'bbox': p_rect, 'class': p_cls})
            
    return true_poison_gts

# --- Part 1.5: 新增模型预测封装函数 ---

def model_predict(model, img):
    """
    封装 YOLO 预测逻辑。
    将 YOLO 的 Results 对象转换为简单的 numpy 数组：
    [[x1, y1, x2, y2, conf, cls], ...]
    """
    # verbose=False 防止打印大量日志
    results = model(img, verbose=False) 
    
    # results[0] 是第一张图的结果
    # .boxes.data 包含了 [x1, y1, x2, y2, conf, cls]
    if len(results) > 0 and len(results[0].boxes) > 0:
        # 转移到 CPU 并转为 numpy
        det = results[0].boxes.data.cpu().numpy()
        return det
    else:
        return []

def get_all_conv_layers(model):
    """获取模型中所有 Conv2d 层及其名称，按前向传播顺序排序"""
    layers = []
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers.append((name, module))
    return layers

def scale_weights_cumulative(model, scale_factor, num_layers_from_back):
    """
    只放大模型最后 num_layers_from_back 个卷积层。
    """
    all_layers = get_all_conv_layers(model)
    total_layers = len(all_layers)
    
    # 确定要修改哪些层 (切片: 取最后 N 个)
    # 如果 num_layers_from_back = 1，取最后一个
    # 如果 num_layers_from_back = 5，取最后5个
    if num_layers_from_back > total_layers:
        num_layers_from_back = total_layers
        
    layers_to_scale = all_layers[-num_layers_from_back:]
    
    backup_params = {}
    modified_count = 0
    
    for name, module in layers_to_scale:
        # 1. 备份
        backup_params[name] = {
            'weight': module.weight.data.clone(),
            'bias': module.bias.data.clone() if module.bias is not None else None
        }
        
        # 2. 放大
        module.weight.data *= scale_factor
        if module.bias is not None:
            module.bias.data *= scale_factor
            
        modified_count += 1
            
    # print(f"DEBUG: 已放大最后 {modified_count} / {total_layers} 个卷积层 (Scale={scale_factor})")
    return backup_params

# 恢复函数的逻辑不变，它会自动根据 backup_params 里的名字来恢复

def restore_weights_universal(model, backup_params):
    """恢复参数"""
    for name, module in model.model.named_modules():
        if name in backup_params:
            if isinstance(module, nn.Conv2d):
                module.weight.data = backup_params[name]['weight']
                if backup_params[name]['bias'] is not None:
                    module.bias.data = backup_params[name]['bias']
# --- Part 2: 修改后的主实验函数 ---

def calculate_consistency_single_img(model, img, boxes_base, iou_thresh=0.5):
    """
    计算单张图片在参数放大后的一致性。
    返回: (匹配上的框数量, 总框数量)
    """
    if len(boxes_base) == 0:
        return 0, 0
        
    # 预测 (此时模型已经是修改过的状态)
    boxes_processed = model_predict(model, img)
    
    matched_count = 0
    total_count = len(boxes_base)
    
    for base_box in boxes_base:
        pred_rect = base_box[:4]
        pred_cls = int(base_box[5])
        
        # 在处理后的结果里找匹配
        is_matched = False
        for cand in boxes_processed:
            cand_cls = int(cand[5])
            cand_rect = cand[:4]
            # 只要类别一致且IoU够大，就算“存活”
            if (cand_cls == pred_cls) and (compute_iou(pred_rect, cand_rect) > iou_thresh):
                is_matched = True
                break
        
        if is_matched:
            matched_count += 1
            
    return matched_count, total_count

def adaptive_layer_selection(model, clean_image_paths, scale_factor=1.1, threshold_xi=0.35):
    """
    【核心算法 Algorithm 1】
    使用一小批纯净验证集，自动搜索最佳层数 k。
    
    参数:
        clean_image_paths: 纯净验证图片的路径列表 (建议 20-50 张)
        scale_factor: 放大系数 (建议使用你之前测试效果最好的 1.1)
        threshold_xi: 错误率阈值 (Error Rate Threshold). 
                      比如 0.35 意味着我们要找那个让 Clean Consistency 降到 0.65 以下的时刻。
    返回:
        optimal_k: 最佳层数
    """
    print(f"\n[Calibration] 正在自动搜索最佳层数 k (Scale={scale_factor}, Error Thresh={threshold_xi})...")
    
    all_layers = get_all_conv_layers(model)
    total_layers = len(all_layers)
    print(f"模型总共有 {total_layers} 个卷积层。")
    
    # 预先计算所有验证图片的基准预测 (Base Prediction)，节省时间
    print("正在预计算基准结果...")
    base_data = [] # 存 (img, boxes_base)
    for p in clean_image_paths:
        img = cv2.imread(p)
        if img is None: continue
        preds = model_predict(model, img)
        if len(preds) > 0:
            base_data.append((img, preds))
            
    if len(base_data) == 0:
        print("Error: 验证集没有检测到任何目标，无法校准！")
        return total_layers

    # 搜索策略：从后向前，步长为 5 (为了快)，你也可以设为 1
    # 根据你的实验，前 30 层几乎没影响，我们可以从 30 层开始搜
    search_range = range(30, total_layers + 1, 5) 
    
    optimal_k = total_layers # 默认备选
    
    for k in search_range:
        # 1. 修改模型 (放大最后 k 层)
        backup = scale_weights_cumulative(model, scale_factor, k)
        
        total_matched = 0
        total_objects = 0
        
        # 2. 跑一遍验证集
        for img, boxes_base in base_data:
            m, t = calculate_consistency_single_img(model, img, boxes_base)
            total_matched += m
            total_objects += t
            
        # 3. 恢复模型 (必须!)
        restore_weights_universal(model, backup)
        
        # 4. 计算指标
        # Consistency = 存活数 / 总数
        current_consistency = total_matched / (total_objects + 1e-6)
        # Error Rate = 1 - Consistency
        error_rate = 1.0 - current_consistency
        
        print(f"  k={k}: Clean Consistency={current_consistency:.4f}, Error Rate={error_rate:.4f}")
        
        # 5. 终止条件
        # 如果错误率超过了阈值 (即干净样本开始崩了)，就是这里！
        if error_rate > threshold_xi:
            print(f"  -> 触发阈值! 干净样本性能已下降。选定 k={k}")
            optimal_k = k
            break
            
    return optimal_k

# --- Part 4: 最终检测流程 ---

def run_final_detection(model, test_image_paths, clean_val_paths, CLEAN_LABEL_DIR, POISON_LABEL_DIR):
    """
    全自动流程：校准 -> 检测
    """
    # ==========================
    # Step 1: 校准 (Calibration)
    # ==========================
    # 参数设置建议：
    # Scale=1.1 (你测出的最佳放大倍数)
    # Threshold=0.35 (对应 Clean Consistency 0.65 左右，根据你 scale 80 层的实验结果设定的)
    BEST_SCALE = 1.1
    ERROR_THRESH = 0.35
    
    optimal_k = adaptive_layer_selection(model, clean_val_paths, 
                                         scale_factor=BEST_SCALE, 
                                         threshold_xi=ERROR_THRESH)
    
    print("="*50)
    print(f"校准完成！后续检测将使用: Scale={BEST_SCALE}, Modify Last {optimal_k} Layers")
    print("="*50)
    
    # ==========================
    # Step 2: 检测 (Detection)
    # ==========================
    print(f"开始对 {len(test_image_paths)} 张未知图片进行检测...")
    
    # 统计最终性能
    stats = {'c_m': 0, 'c_t': 0, 'p_m': 0, 'p_t': 0}
    
    for i, img_path in enumerate(test_image_paths):
        if i % 20 == 0: print(f"Processing {i}...")
        
        # 路径处理
        img_name = os.path.basename(img_path)
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        poison_label_path = os.path.join(POISON_LABEL_DIR, label_name)
        clean_label_path = os.path.join(CLEAN_LABEL_DIR, label_name)

        img0 = cv2.imread(img_path)
        if img0 is None: continue
        h0, w0 = img0.shape[:2]

        # 获取真值 (仅用于最后算分，实际检测逻辑不利用此信息)
        poison_instances = get_poisoned_gt_boxes_blind(clean_label_path, poison_label_path, h0, w0)
        
        # 1. 基准预测
        boxes_base = model_predict(model, img0)
        if len(boxes_base) == 0: continue
        
        # 2. 干扰预测 (只修改 optimal_k 层)
        backup = scale_weights_cumulative(model, BEST_SCALE, optimal_k)
        boxes_processed = model_predict(model, img0)
        restore_weights_universal(model, backup)
        
        # 3. 逐框判断
        for base_box in boxes_base:
            pred_rect = base_box[:4]
            pred_cls = int(base_box[5])
            
            # --- 检测逻辑 (The Detector) ---
            # 检查这个框在干扰后还在不在
            is_consistent = False
            for cand in boxes_processed:
                if (int(cand[5]) == pred_cls) and (compute_iou(pred_rect, cand[:4]) > 0.5):
                    is_consistent = True
                    break
            
            # 判定结果
            # 如果 一致性高 (Consistent) -> 判定为毒 (Poison)
            # 如果 一致性低 (Inconsistent) -> 判定为干净 (Clean)
            detected_as_poison = is_consistent 
            
            # --- 评估逻辑 (Evaluation) ---
            # 看看实际上是不是毒 (Ground Truth)
            is_actually_poison = False
            for gt_item in poison_instances:
                if (compute_iou(pred_rect, gt_item['bbox']) > 0.5) and (pred_cls == gt_item['class']):
                    is_actually_poison = True
                    break
            
            # 记录混淆矩阵数据
            if is_actually_poison:
                stats['p_t'] += 1 # 实际是毒
                if detected_as_poison: stats['p_m'] += 1 # 成功检测出毒 (True Positive)
            else:
                stats['c_t'] += 1 # 实际是干净
                if not detected_as_poison: stats['c_m'] += 1 # 成功判定为干净 (True Negative)
                # 注意：这里 stats['c_m'] 定义变了，代表“正确防御的干净样本”
                # 之前代码里 c_m 代表“一致性高的干净样本”(即检测失败的)
                # 为了不混淆，我们看下面的打印逻辑

    # ==========================
    # Step 3: 输出结果
    # ==========================
    # 毒样本召回率 (TPR): 有多少毒样本因为“过于稳固”而被抓出来了？
    tpr = stats['p_m'] / (stats['p_t'] + 1e-6)
    
    # 干净样本保留率 (TN/Total Clean): 有多少干净样本因为“变弱了”而被正确放行了？
    # 注意检测逻辑：detected_as_poison = is_consistent
    # 所以干净样本我们要看的是 !is_consistent 的比例
    tnr = 1.0 - (stats['c_t'] - stats['c_m']) / (stats['c_t'] + 1e-6) 
    # 修正一下统计逻辑以便理解：
    # 之前循环里: if not detected_as_poison: stats['c_m'] += 1
    # 所以 stats['c_m'] 就是 TN
    tnr = stats['c_m'] / (stats['c_t'] + 1e-6)

    print("\n=== 最终检测报告 ===")
    print(f"使用的参数: Scale={BEST_SCALE}, k={optimal_k}")
    print(f"毒样本检测成功率 (TPR): {tpr:.4f} (期望越高越好，比如 > 0.9)")
    print(f"干净样本误报率 (FPR):   {1-tnr:.4f} (期望越低越好，< 0.1)")
    print(f"干净样本保留率 (TNR):   {tnr:.4f}")



# --- Part 3: 主函数入口 (必须配置这里) ---

if __name__=='__main__':
    # 1. 设置路径 (请修改为你实际的路径)
    # 存放中毒图片的文件夹
    POISON_IMG_DIR = "D://BaiduNetdiskDownload//Poisoned_dataset//Poisoned_Dataset_Pack//BadNets//poisoned_badnets//images//train" 
    # 存放纯净标签的文件夹
    CLEAN_LABEL_DIR = "D://BaiduNetdiskDownload//Poisoned_dataset//clean//labels//train"
    # 存放中毒标签的文件夹
    POISON_LABEL_DIR = "D://BaiduNetdiskDownload//Poisoned_dataset//Poisoned_Dataset_Pack//BadNets//poisoned_badnets//labels//train"
    # 你的模型路径 (例如 best.pt)
    MODEL_PATH = "D://BaiduNetdiskDownload//Poisoned_dataset//best.pt" 

    # 2. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # 3. 获取所有图片路径
    # 支持 jpg 和 png
    image_paths = glob.glob(os.path.join(POISON_IMG_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(POISON_IMG_DIR, "*.png"))
    
    clean_val_paths = glob.glob(os.path.join(CLEAN_LABEL_DIR, "*.jpg"))[:50]
    
    # 2. 拿剩下的混合图片做测试
    test_paths = glob.glob(os.path.join(POISON_IMG_DIR, "*.jpg"))[:200]
    
    run_final_detection(model, test_paths, clean_val_paths, CLEAN_LABEL_DIR, POISON_LABEL_DIR)