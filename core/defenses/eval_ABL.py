import os

def load_file_names(file_path):
    """
    读取文件列表，提取纯文件名（去除路径），用于比对
    """
    file_names = set()
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return file_names
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 1. 有些 poison_list 格式是 "path label"，需要分割
                # 2. 使用 os.path.basename 只取文件名 (如 '0001.jpg')，忽略文件夹差异
                clean_name = os.path.basename(line.split()[0])
                file_names.add(clean_name)
    except Exception as e:
        print(f"[Error] Reading {file_path} failed: {e}")
        
    return file_names

def evaluate_abl_performance(abl_result_path, ground_truth_path):
    print("="*60)
    print(">>> ABL Isolation Performance Evaluation")
    print("="*60)

    # 1. 加载数据
    print(f"Loading ABL isolated list: {abl_result_path}")
    abl_set = load_file_names(abl_result_path)
    
    print(f"Loading Ground Truth list: {ground_truth_path}")
    gt_set = load_file_names(ground_truth_path)

    if len(abl_set) == 0 or len(gt_set) == 0:
        print("[Error] One of the lists is empty. Check your paths.")
        return

    # 2. 计算指标
    # TP (True Positive): ABL 挑出来的，且确实在后门列表里的 (抓对了)
    tp_set = abl_set.intersection(gt_set)
    TP = len(tp_set)

    # FP (False Positive): ABL 挑出来的，但不在后门列表里的 (误伤了干净样本)
    fp_set = abl_set - gt_set
    FP = len(fp_set)

    # FN (False Negative): 在后门列表里，但 ABL 没挑出来的 (漏网之鱼)
    fn_set = gt_set - abl_set
    FN = len(fn_set)

    # 3. 计算比率
    # Precision (查准率): 挑出来的样本里，有多少是真的后门？
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Recall (查全率/TPR): 所有后门样本里，抓出来了多少？
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # F1 Score: 综合指标
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 4. 打印报告
    print("\n" + "-"*30)
    print("       Evaluation Report")
    print("-"*30)
    print(f"Total Isolated by ABL: {len(abl_set)}")
    print(f"Total Real Poisoned:   {len(gt_set)}")
    print("-" * 30)
    print(f"TP (Caught Correctly): {TP}")
    print(f"FP (Wrongly Accused):  {FP}  <-- Clean images treated as poison")
    print(f"FN (Missed Poison):    {FN}  <-- Poison images missed")
    print("-" * 30)
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f}    ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60)

if __name__ == "__main__":
    # 配置你的路径
    abl_path = "D://BaiduNetdiskDownload//Poisoned_dataset//Poisoned_Dataset_Pack//BadNets//abl_isolated_samples.txt"
    gt_path = "D://BaiduNetdiskDownload//Poisoned_dataset//Poisoned_Dataset_Pack//BadNets//poisoned_badnets//poison_list.txt"
    
    evaluate_abl_performance(abl_result_path=abl_path, ground_truth_path=gt_path)