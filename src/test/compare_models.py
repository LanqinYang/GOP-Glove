#!/usr/bin/env python3
"""
多模型统计比较脚本
用于比较不同模型的LOSO交叉验证结果，进行统计显著性检验

使用方法:
    python compare_models.py --models ADANN LightGBM ADANN_LightGBM
    python compare_models.py --results_dir outputs/ --models ADANN LightGBM
    python compare_models.py --manual --model1 "ADANN" --scores1 0.85,0.87,0.83 --model2 "LightGBM" --scores2 0.82,0.84,0.81
"""

import argparse
import json
import glob
import os
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.evaluation.statistical_tests import ModelComparisonStatistics, compare_loso_results

def load_loso_results(results_dir: str, model_names: list) -> dict:
    """
    从输出目录加载LOSO结果
    
    Args:
        results_dir: 结果目录路径
        model_names: 模型名称列表
        
    Returns:
        dict: {model_name: accuracies_array}
    """
    model_scores = {}
    
    for model_name in model_names:
        # 查找对应的统计文件
        pattern = os.path.join(results_dir, model_name, "loso", "*", f"loso_stats_{model_name}.json")
        stat_files = glob.glob(pattern)
        
        if not stat_files:
            print(f"⚠️ 未找到模型 {model_name} 的LOSO统计结果")
            print(f"   搜索路径: {pattern}")
            continue
        
        # 使用最新的结果文件
        latest_file = max(stat_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            fold_accuracies = data['fold_accuracies']
            model_scores[model_name] = np.array(fold_accuracies)
            
            print(f"✅ 加载 {model_name} LOSO结果: {len(fold_accuracies)} folds")
            print(f"   平均准确率: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
            
        except Exception as e:
            print(f"❌ 加载 {model_name} 结果失败: {e}")
    
    return model_scores

def manual_input_scores(model1_name: str, scores1_str: str, 
                       model2_name: str, scores2_str: str) -> dict:
    """
    手动输入分数进行比较
    
    Args:
        model1_name: 模型1名称
        scores1_str: 模型1分数，逗号分隔
        model2_name: 模型2名称  
        scores2_str: 模型2分数，逗号分隔
        
    Returns:
        dict: {model_name: accuracies_array}
    """
    try:
        scores1 = np.array([float(x.strip()) for x in scores1_str.split(',')])
        scores2 = np.array([float(x.strip()) for x in scores2_str.split(',')])
        
        return {
            model1_name: scores1,
            model2_name: scores2
        }
    except ValueError as e:
        print(f"❌ 分数解析失败: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='多模型LOSO结果统计比较')
    
    # 主要模式选择
    parser.add_argument('--manual', action='store_true', 
                       help='手动输入模式，直接提供分数进行比较')
    
    # 自动加载模式参数
    parser.add_argument('--results_dir', type=str, default='outputs',
                       help='结果输出目录路径 (默认: outputs)')
    parser.add_argument('--models', nargs='+', 
                       choices=['1D_CNN', 'XGBoost', 'RAC', 'Transformer_Encoder', 
                               'ADANN', 'ADANN_LightGBM', 'LightGBM'],
                       help='要比较的模型列表')
    
    # 手动输入模式参数
    parser.add_argument('--model1', type=str, help='模型1名称 (手动模式)')
    parser.add_argument('--scores1', type=str, help='模型1分数，逗号分隔 (手动模式)')
    parser.add_argument('--model2', type=str, help='模型2名称 (手动模式)')
    parser.add_argument('--scores2', type=str, help='模型2分数，逗号分隔 (手动模式)')
    
    # 统计参数
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='显著性水平 (默认: 0.05)')
    parser.add_argument('--no_plot', action='store_true',
                       help='不生成可视化图表')
    
    args = parser.parse_args()
    
    print("🔬 多模型LOSO结果统计比较分析")
    print("=" * 60)
    
    # 获取模型分数
    model_scores = {}
    
    if args.manual:
        # 手动输入模式
        if not all([args.model1, args.scores1, args.model2, args.scores2]):
            print("❌ 手动模式需要提供 --model1, --scores1, --model2, --scores2 参数")
            return 1
        
        model_scores = manual_input_scores(args.model1, args.scores1, 
                                         args.model2, args.scores2)
        
        if not model_scores:
            return 1
    
    else:
        # 自动加载模式
        if not args.models:
            print("❌ 请提供要比较的模型列表 --models")
            return 1
        
        if len(args.models) < 2:
            print("❌ 至少需要提供2个模型进行比较")
            return 1
        
        model_scores = load_loso_results(args.results_dir, args.models)
        
        if len(model_scores) < 2:
            print("❌ 成功加载的模型少于2个，无法进行比较")
            return 1
    
    # 进行统计比较
    print(f"\n🎯 开始统计比较分析 (α = {args.alpha})...")
    
    try:
        # 创建统计检验实例
        tester = ModelComparisonStatistics(alpha=args.alpha)
        
        # 进行综合比较
        results = tester.comprehensive_comparison(
            model_scores, 
            plot_results=not args.no_plot
        )
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"model_comparison_results_{timestamp}.json"
        
        # 转换numpy数组和numpy标量为Python原生类型以便JSON序列化
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_results = convert_numpy_types(results)
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 比较结果已保存至: {output_file}")
        
        # 生成简化的结论报告
        print(f"\n📋 简化结论报告:")
        print("=" * 40)
        
        # 找出最佳模型
        means = {name: np.mean(scores) for name, scores in model_scores.items()}
        best_model = max(means.items(), key=lambda x: x[1])
        
        print(f"🏆 最佳模型: {best_model[0]} (准确率: {best_model[1]:.4f})")
        
        # 统计显著差异
        significant_count = sum(1 for result in results['pairwise_comparisons'].values() 
                              if result.get('significant', False))
        total_comparisons = len(results['pairwise_comparisons'])
        
        print(f"🔬 显著差异: {significant_count}/{total_comparisons} 对比较有统计显著性")
        
        if significant_count > 0:
            print("✅ 建议: 报告统计检验结果，支持模型选择的科学性")
        else:
            print("⚠️ 建议: 无显著差异时，可考虑计算复杂度、部署便利性等因素")
        
        return 0
        
    except Exception as e:
        print(f"❌ 统计分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    from datetime import datetime
    sys.exit(main())