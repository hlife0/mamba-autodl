import os
import csv
import argparse
from glob import glob
import re
from collections import defaultdict

def normalize_answer(text):
    """
    标准化答案：去除标点、多余空格，转小写
    """
    if not text:
        return ""
    # 转小写
    text = text.lower().strip()
    # 去除标点
    text = re.sub(r'[^\w\s]', ' ', text)
    # 去除多余空格
    text = ' '.join(text.split())
    return text

def exact_match(prediction, ground_truth):
    """
    检查预测是否与真实答案完全匹配（标准化后）
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def contains_match(prediction, ground_truth):
    """
    检查预测中是否包含真实答案（更宽松的匹配）
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    return gt_norm in pred_norm if gt_norm else False

def analyze_csv(csv_file):
    """
    分析单个CSV文件的准确率
    
    Returns:
        dict: 包含统计信息的字典
    """
    total = 0
    exact_correct = 0
    contain_correct = 0
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            prediction = row.get('decoded', '')
            ground_truth = row.get('answer', '')
            
            if exact_match(prediction, ground_truth):
                exact_correct += 1
                contain_correct += 1
            elif contains_match(prediction, ground_truth):
                contain_correct += 1
    
    return {
        'total': total,
        'exact_correct': exact_correct,
        'contain_correct': contain_correct,
        'exact_accuracy': exact_correct / total * 100 if total > 0 else 0,
        'contain_accuracy': contain_correct / total * 100 if total > 0 else 0
    }

def find_latest_csv(experiments_dir, experiment_name):
    """
    找到指定实验名称的最新CSV文件
    
    Args:
        experiments_dir: 实验目录
        experiment_name: 实验名称（如 'baseline_empty', 'skiplayers30'）
    
    Returns:
        str or None: 最新的CSV文件路径，如果不存在返回None
    """
    # 构建模式匹配
    pattern = os.path.join(experiments_dir, f"{experiment_name}_*.csv")
    matching_files = glob(pattern)
    
    if not matching_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_file = max(matching_files, key=os.path.getmtime)
    return latest_file

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='分析实验结果的准确率',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python analyse_result.py baseline_empty baseline_doc1 baseline_doc2 baseline_full
  python analyse_result.py skiplayers30 skiplayers20 skiplayers40
        """
    )
    parser.add_argument('experiments', nargs='+', 
                        help='实验名称列表 (如 baseline_empty, skiplayers30, etc.)')
    parser.add_argument('--dir', type=str, default='./skip_layer_pre/experiments',
                        help='实验结果目录 (默认: ./skip_layer_pre/experiments)')
    
    args = parser.parse_args()
    
    experiments_dir = args.dir
    experiment_names = args.experiments
    
    # 查找并分析每个实验
    print("=" * 100)
    print("Experiment Results Analysis")
    print("=" * 100)
    print()
    
    results_summary = {}
    
    for exp_name in experiment_names:
        latest_file = find_latest_csv(experiments_dir, exp_name)
        
        if latest_file is None:
            print(f"⚠️  {exp_name:<20} - No files found")
            continue
        
        filename = os.path.basename(latest_file)
        results = analyze_csv(latest_file)
        
        print(f"📊 {exp_name.upper()}")
        print(f"   File: {filename}")
        print(f"   Total samples: {results['total']}")
        print(f"   Exact Match: {results['exact_correct']}/{results['total']} = {results['exact_accuracy']:.2f}%")
        print(f"   Contain Match: {results['contain_correct']}/{results['total']} = {results['contain_accuracy']:.2f}%")
        print()
        
        results_summary[exp_name] = {
            'file': filename,
            'results': results
        }
    
    # 打印摘要对比表
    if results_summary:
        print("=" * 100)
        print("Summary Comparison")
        print("=" * 100)
        print()
        print(f"{'Experiment':<25} {'Samples':<10} {'Exact Match':<20} {'Contain Match':<20}")
        print("-" * 100)
        
        for exp_name in experiment_names:
            if exp_name in results_summary:
                data = results_summary[exp_name]
                results = data['results']
                
                print(f"{exp_name:<25} {results['total']:<10} "
                      f"{results['exact_correct']:>4}/{results['total']:<4} ({results['exact_accuracy']:>5.2f}%)   "
                      f"{results['contain_correct']:>4}/{results['total']:<4} ({results['contain_accuracy']:>5.2f}%)")
        
        print("=" * 100)
        print()

if __name__ == "__main__":
    main()
