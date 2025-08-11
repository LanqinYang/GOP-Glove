"""
统计显著性检验模块
用于机器学习模型性能比较的统计检验方法

主要包含：
1. Wilcoxon符号秩检验 (非参数检验)
2. 配对t检验 (参数检验)
3. McNemar检验 (分类错误比较)
4. Friedman检验 (多模型比较)
5. 效应量计算 (Cohen's d, Cliff's delta)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, friedmanchisquare
import warnings
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparisonStatistics:
    """模型比较统计分析类"""
    
    def __init__(self, alpha=0.05):
        """
        初始化统计检验
        
        Args:
            alpha: 显著性水平，默认0.05
        """
        self.alpha = alpha
        self.results = {}
    
    def wilcoxon_signed_rank_test(self, model1_scores: np.ndarray, model2_scores: np.ndarray, 
                                  model1_name: str = "Model1", model2_name: str = "Model2") -> Dict:
        """
        Wilcoxon符号秩检验 - 比较两个相关样本
        
        适用场景：
        - LOSO交叉验证结果比较
        - 同一数据集上不同模型性能比较
        - 非参数检验，不要求数据正态分布
        
        Args:
            model1_scores: 模型1的性能分数 (如每个fold的准确率)
            model2_scores: 模型2的性能分数
            model1_name: 模型1名称
            model2_name: 模型2名称
            
        Returns:
            Dict: 包含检验统计量、p值、效应量等信息
        """
        print(f"\n🔬 Wilcoxon符号秩检验: {model1_name} vs {model2_name}")
        print("=" * 60)
        
        # 检查输入
        if len(model1_scores) != len(model2_scores):
            raise ValueError("两个模型的分数数量必须相同")
        
        if len(model1_scores) < 6:
            warnings.warn("样本量较小 (n<6)，检验结果可能不可靠")
        
        # 计算差值
        differences = model1_scores - model2_scores
        
        # 基本统计信息
        print(f"📊 基本统计信息:")
        print(f"   {model1_name}: {np.mean(model1_scores):.4f} ± {np.std(model1_scores):.4f}")
        print(f"   {model2_name}: {np.mean(model2_scores):.4f} ± {np.std(model2_scores):.4f}")
        print(f"   平均差值: {np.mean(differences):.4f} ± {np.std(differences):.4f}")
        
        # Wilcoxon检验
        try:
            if np.all(differences == 0):
                print("⚠️  所有差值为0，无法进行检验")
                return {
                    'test_name': 'Wilcoxon Signed-Rank Test',
                    'statistic': np.nan,
                    'p_value': 1.0,
                    'significant': False,
                    'effect_size': 0.0,
                    'interpretation': '两模型性能完全相同'
                }
                
            statistic, p_value = wilcoxon(differences, alternative='two-sided')
            
            # 效应量 (r = Z / sqrt(N))
            n = len(differences)
            z_score = stats.norm.ppf(1 - p_value/2)  # 近似Z分数
            effect_size = z_score / np.sqrt(n)
            
            # Cliff's delta (更适合非参数检验的效应量)
            cliffs_delta = self._calculate_cliffs_delta(model1_scores, model2_scores)
            
        except ValueError as e:
            print(f"❌ 检验失败: {e}")
            return {
                'test_name': 'Wilcoxon Signed-Rank Test',
                'error': str(e)
            }
        
        # 结果解释
        is_significant = p_value < self.alpha
        
        print(f"\n🔍 检验结果:")
        print(f"   检验统计量: {statistic:.4f}")
        print(f"   p值: {p_value:.6f}")
        print(f"   显著性水平: {self.alpha}")
        print(f"   是否显著: {'是' if is_significant else '否'}")
        print(f"   效应量 (r): {effect_size:.4f}")
        print(f"   Cliff's Delta: {cliffs_delta:.4f}")
        
        # 效应量解释
        effect_interpretation = self._interpret_effect_size(abs(cliffs_delta), 'cliffs_delta')
        print(f"   效应量大小: {effect_interpretation}")
        
        # 统计结论
        if is_significant:
            better_model = model1_name if np.mean(differences) > 0 else model2_name
            print(f"\n✅ 统计结论: {better_model} 显著优于另一模型 (p < {self.alpha})")
        else:
            print(f"\n❌ 统计结论: 两模型性能差异不显著 (p ≥ {self.alpha})")
        
        result = {
            'test_name': 'Wilcoxon Signed-Rank Test',
            'model1_name': model1_name,
            'model2_name': model2_name,
            'n_samples': len(model1_scores),
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'significant': is_significant,
            'effect_size_r': float(effect_size),
            'cliffs_delta': float(cliffs_delta),
            'effect_interpretation': effect_interpretation,
            'mean_difference': float(np.mean(differences)),
            'model1_mean': float(np.mean(model1_scores)),
            'model2_mean': float(np.mean(model2_scores)),
            'better_model': better_model if is_significant else 'No significant difference'
        }
        
        self.results[f"{model1_name}_vs_{model2_name}_wilcoxon"] = result
        return result
    
    def paired_t_test(self, model1_scores: np.ndarray, model2_scores: np.ndarray,
                      model1_name: str = "Model1", model2_name: str = "Model2") -> Dict:
        """
        配对t检验 - 参数检验
        
        适用场景：
        - 数据近似正态分布
        - 样本量较大 (n > 30) 或确认正态性
        
        Args:
            model1_scores: 模型1的性能分数
            model2_scores: 模型2的性能分数
            model1_name: 模型1名称
            model2_name: 模型2名称
            
        Returns:
            Dict: 检验结果
        """
        print(f"\n🔬 配对t检验: {model1_name} vs {model2_name}")
        print("=" * 60)
        
        # 正态性检验
        differences = model1_scores - model2_scores
        _, p_normal = stats.shapiro(differences)
        
        print(f"📊 正态性检验 (Shapiro-Wilk):")
        print(f"   p值: {p_normal:.6f}")
        if p_normal < 0.05:
            print("   ⚠️  差值不服从正态分布，建议使用Wilcoxon检验")
        else:
            print("   ✅ 差值近似正态分布，可以使用t检验")
        
        # t检验
        t_stat, p_value = ttest_rel(model1_scores, model2_scores)
        
        # Cohen's d 效应量
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
        
        # 结果
        is_significant = p_value < self.alpha
        
        print(f"\n🔍 检验结果:")
        print(f"   t统计量: {t_stat:.4f}")
        print(f"   p值: {p_value:.6f}")
        print(f"   Cohen's d: {cohens_d:.4f}")
        
        effect_interpretation = self._interpret_effect_size(abs(cohens_d), 'cohens_d')
        print(f"   效应量大小: {effect_interpretation}")
        
        if is_significant:
            better_model = model1_name if t_stat > 0 else model2_name
            print(f"\n✅ 统计结论: {better_model} 显著优于另一模型 (p < {self.alpha})")
        else:
            print(f"\n❌ 统计结论: 两模型性能差异不显著 (p ≥ {self.alpha})")
        
        result = {
            'test_name': 'Paired t-test',
            'model1_name': model1_name,
            'model2_name': model2_name,
            'normality_p': float(p_normal),
            'normality_ok': p_normal >= 0.05,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': is_significant,
            'cohens_d': float(cohens_d),
            'effect_interpretation': effect_interpretation,
            'better_model': better_model if is_significant else 'No significant difference'
        }
        
        self.results[f"{model1_name}_vs_{model2_name}_ttest"] = result
        return result
    
    def friedman_test(self, model_scores: Dict[str, np.ndarray]) -> Dict:
        """
        Friedman检验 - 多个模型比较 (非参数版本的重复测量ANOVA)
        
        适用场景：
        - 比较3个或更多模型
        - 同一数据集上的多个fold结果
        
        Args:
            model_scores: {model_name: scores_array} 格式的字典
            
        Returns:
            Dict: 检验结果
        """
        print(f"\n🔬 Friedman检验: 多模型比较")
        print("=" * 60)
        
        model_names = list(model_scores.keys())
        scores_arrays = list(model_scores.values())
        
        # 检查输入
        n_models = len(model_names)
        if n_models < 3:
            raise ValueError("Friedman检验需要至少3个模型")
        
        # 检查所有模型的样本数是否一致
        n_samples = [len(scores) for scores in scores_arrays]
        if len(set(n_samples)) > 1:
            raise ValueError("所有模型的样本数必须相同")
        
        n_samples = n_samples[0]
        
        print(f"📊 比较信息:")
        print(f"   模型数量: {n_models}")
        print(f"   样本数量: {n_samples}")
        
        for name, scores in model_scores.items():
            print(f"   {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        # Friedman检验
        statistic, p_value = friedmanchisquare(*scores_arrays)
        
        is_significant = p_value < self.alpha
        
        print(f"\n🔍 检验结果:")
        print(f"   Friedman统计量: {statistic:.4f}")
        print(f"   p值: {p_value:.6f}")
        print(f"   是否显著: {'是' if is_significant else '否'}")
        
        if is_significant:
            print(f"\n✅ 统计结论: 模型之间存在显著差异 (p < {self.alpha})")
            print("   建议进行事后检验 (post-hoc) 确定具体差异")
        else:
            print(f"\n❌ 统计结论: 模型之间无显著差异 (p ≥ {self.alpha})")
        
        result = {
            'test_name': 'Friedman Test',
            'model_names': model_names,
            'n_models': n_models,
            'n_samples': n_samples,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': is_significant,
            'model_means': {name: float(np.mean(scores)) for name, scores in model_scores.items()}
        }
        
        self.results['friedman_test'] = result
        return result
    
    def comprehensive_comparison(self, model_scores: Dict[str, np.ndarray], 
                                plot_results: bool = True) -> Dict:
        """
        综合模型比较分析
        
        Args:
            model_scores: {model_name: scores_array} 格式的字典
            plot_results: 是否生成可视化图表
            
        Returns:
            Dict: 完整的比较结果
        """
        print(f"\n🔍 综合模型比较分析")
        print("=" * 80)
        
        model_names = list(model_scores.keys())
        n_models = len(model_names)
        
        results = {
            'summary': {},
            'pairwise_comparisons': {},
            'friedman_test': None
        }
        
        # 1. 描述性统计
        print(f"\n📊 描述性统计:")
        summary_stats = {}
        for name, scores in model_scores.items():
            stats_dict = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'n': len(scores)
            }
            summary_stats[name] = stats_dict
            print(f"   {name:15}: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f} "
                  f"(范围: {stats_dict['min']:.4f}-{stats_dict['max']:.4f})")
        
        results['summary'] = summary_stats
        
        # 2. Friedman检验 (如果模型数 ≥ 3)
        if n_models >= 3:
            friedman_result = self.friedman_test(model_scores)
            results['friedman_test'] = friedman_result
        
        # 3. 两两比较
        print(f"\n🔬 两两比较 (Wilcoxon符号秩检验):")
        pairwise_results = {}
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model1 = model_names[i]
                model2 = model_names[j]
                
                wilcoxon_result = self.wilcoxon_signed_rank_test(
                    model_scores[model1], model_scores[model2], model1, model2
                )
                
                pairwise_results[f"{model1}_vs_{model2}"] = wilcoxon_result
        
        results['pairwise_comparisons'] = pairwise_results
        
        # 4. 生成可视化
        if plot_results:
            self._plot_comparison_results(model_scores, results)
        
        # 5. 生成总结报告
        self._generate_summary_report(results)
        
        return results
    
    def _calculate_cliffs_delta(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算Cliff's delta效应量"""
        n1, n2 = len(x), len(y)
        delta = 0
        
        for i in range(n1):
            for j in range(n2):
                if x[i] > y[j]:
                    delta += 1
                elif x[i] < y[j]:
                    delta -= 1
        
        return delta / (n1 * n2)
    
    def _interpret_effect_size(self, effect_size: float, effect_type: str) -> str:
        """解释效应量大小"""
        if effect_type == 'cohens_d':
            if effect_size < 0.2:
                return "negligible (可忽略)"
            elif effect_size < 0.5:
                return "small (小)"
            elif effect_size < 0.8:
                return "medium (中等)"
            else:
                return "large (大)"
        
        elif effect_type == 'cliffs_delta':
            if effect_size < 0.147:
                return "negligible (可忽略)"
            elif effect_size < 0.33:
                return "small (小)"
            elif effect_size < 0.474:
                return "medium (中等)"
            else:
                return "large (大)"
        
        return "unknown"
    
    def _plot_comparison_results(self, model_scores: Dict[str, np.ndarray], 
                                results: Dict) -> None:
        """生成比较结果的可视化图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Comparison Analysis', fontsize=16)
            
            # 1. Box plot
            ax1 = axes[0, 0]
            data_for_plot = [scores for scores in model_scores.values()]
            labels = list(model_scores.keys())
            
            ax1.boxplot(data_for_plot, labels=labels)
            ax1.set_title('Performance Distribution')
            ax1.set_ylabel('Accuracy')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Mean comparison with error bars
            ax2 = axes[0, 1]
            means = [np.mean(scores) for scores in model_scores.values()]
            stds = [np.std(scores) for scores in model_scores.values()]
            
            bars = ax2.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)
            ax2.set_title('Mean Performance with Standard Deviation')
            ax2.set_ylabel('Accuracy')
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, mean in zip(bars, means):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{mean:.3f}', ha='center', va='bottom')
            
            # 3. P-value heatmap for pairwise comparisons
            ax3 = axes[1, 0]
            if len(results['pairwise_comparisons']) > 0:
                model_names = list(model_scores.keys())
                n_models = len(model_names)
                p_matrix = np.ones((n_models, n_models))
                
                for key, result in results['pairwise_comparisons'].items():
                    model1, model2 = key.split('_vs_')
                    i = model_names.index(model1)
                    j = model_names.index(model2)
                    p_val = result['p_value']
                    p_matrix[i, j] = p_val
                    p_matrix[j, i] = p_val
                
                im = ax3.imshow(p_matrix, cmap='RdYlBu_r', aspect='auto')
                ax3.set_xticks(range(n_models))
                ax3.set_yticks(range(n_models))
                ax3.set_xticklabels(model_names, rotation=45)
                ax3.set_yticklabels(model_names)
                ax3.set_title('P-values (Wilcoxon Test)')
                
                # 添加p值文本
                for i in range(n_models):
                    for j in range(n_models):
                        if i != j:
                            text = ax3.text(j, i, f'{p_matrix[i, j]:.3f}',
                                          ha="center", va="center", color="black")
                
                plt.colorbar(im, ax=ax3)
            
            # 4. Effect size comparison
            ax4 = axes[1, 1]
            if len(results['pairwise_comparisons']) > 0:
                comparisons = []
                effect_sizes = []
                
                for key, result in results['pairwise_comparisons'].items():
                    comparisons.append(key.replace('_vs_', '\nvs\n'))
                    effect_sizes.append(abs(result['cliffs_delta']))
                
                bars = ax4.bar(range(len(comparisons)), effect_sizes, alpha=0.7)
                ax4.set_title("Effect Sizes (|Cliff's Delta|)")
                ax4.set_ylabel("Effect Size")
                ax4.set_xticks(range(len(comparisons)))
                ax4.set_xticklabels(comparisons, rotation=0, fontsize=8)
                
                # 添加效应量解释线
                ax4.axhline(y=0.147, color='green', linestyle='--', alpha=0.7, label='Small')
                ax4.axhline(y=0.33, color='orange', linestyle='--', alpha=0.7, label='Medium')
                ax4.axhline(y=0.474, color='red', linestyle='--', alpha=0.7, label='Large')
                ax4.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")
    
    def _generate_summary_report(self, results: Dict) -> None:
        """生成总结报告"""
        print(f"\n📋 统计分析总结报告")
        print("=" * 80)
        
        # 模型排名
        means = {name: stats['mean'] for name, stats in results['summary'].items()}
        ranked_models = sorted(means.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 模型性能排名:")
        for i, (model, mean_score) in enumerate(ranked_models, 1):
            print(f"   {i}. {model}: {mean_score:.4f}")
        
        # 显著性差异总结
        print(f"\n🔬 显著性差异总结:")
        significant_pairs = []
        for key, result in results['pairwise_comparisons'].items():
            if result['significant']:
                significant_pairs.append({
                    'comparison': key,
                    'p_value': result['p_value'],
                    'better_model': result['better_model'],
                    'effect_size': result['cliffs_delta']
                })
        
        if significant_pairs:
            print(f"   发现 {len(significant_pairs)} 对模型存在显著差异:")
            for pair in significant_pairs:
                print(f"   • {pair['comparison']}: {pair['better_model']} 更优 "
                      f"(p={pair['p_value']:.6f}, δ={pair['effect_size']:.3f})")
        else:
            print("   ❌ 没有发现显著的模型性能差异")
        
        # 建议
        print(f"\n📝 统计分析建议:")
        if len(ranked_models) > 0:
            best_model = ranked_models[0][0]
            print(f"   • 最佳模型: {best_model}")
            
            if significant_pairs:
                print(f"   • 存在显著差异，建议选择统计显著优于其他模型的方案")
            else:
                print(f"   • 无显著差异，可根据其他因素(如计算复杂度、部署难度)选择模型")
                
            print(f"   • 建议报告完整的统계分析结果，包括p值和效应量")
            print(f"   • 对于论文发表，Wilcoxon检验结果比简单准确率比较更有说服力")

# 便利函数
def quick_wilcoxon_test(model1_scores: np.ndarray, model2_scores: np.ndarray,
                       model1_name: str = "Model1", model2_name: str = "Model2",
                       alpha: float = 0.05) -> Dict:
    """快速进行Wilcoxon符号秩检验"""
    tester = ModelComparisonStatistics(alpha=alpha)
    return tester.wilcoxon_signed_rank_test(model1_scores, model2_scores, model1_name, model2_name)

def compare_loso_results(loso_results: Dict[str, np.ndarray], alpha: float = 0.05) -> Dict:
    """比较LOSO交叉验证结果"""
    tester = ModelComparisonStatistics(alpha=alpha)
    return tester.comprehensive_comparison(loso_results, plot_results=True)

# 示例使用
if __name__ == "__main__":
    # 模拟LOSO结果数据
    np.random.seed(42)
    
    # 假设有6个被试的LOSO结果
    adann_results = np.array([0.85, 0.87, 0.83, 0.86, 0.88, 0.84])  # ADANN在6个fold的准确率
    lightgbm_results = np.array([0.82, 0.84, 0.81, 0.83, 0.85, 0.80])  # LightGBM结果
    hybrid_results = np.array([0.89, 0.91, 0.87, 0.90, 0.92, 0.88])  # ADANN+LightGBM结果
    
    # 进行综合比较
    model_scores = {
        'ADANN': adann_results,
        'LightGBM': lightgbm_results,
        'ADANN+LightGBM': hybrid_results
    }
    
    results = compare_loso_results(model_scores)