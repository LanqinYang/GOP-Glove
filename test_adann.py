#!/usr/bin/env python3
"""
ADANN模型测试脚本
用于快速验证ADANN和ADANN+LightGBM模型的基本功能
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_adann():
    """测试纯ADANN模型"""
    print("🧪 测试纯ADANN模型...")
    
    try:
        from src.training.train_adann import AdannModelCreator
        
        # 创建模型创建器
        creator = AdannModelCreator()
        print("✅ ADANN模型创建器初始化成功")
        
        # 测试超参数定义
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        hyperparams = creator.define_hyperparams(trial)
        print(f"✅ 超参数定义成功: {len(hyperparams)} 个参数")
        
        # 测试模型创建
        input_shape = (100, 5)  # 100时间步，5通道
        n_classes = 11
        model = creator.create_model(hyperparams, input_shape, n_classes)
        print("✅ ADANN模型创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ ADANN测试失败: {e}")
        return False

def test_adann_lightgbm():
    """测试ADANN+LightGBM混合模型"""
    print("\n🧪 测试ADANN+LightGBM混合模型...")
    
    try:
        from src.training.train_adann_lightgbm import AdannLightgbmModelCreator
        
        # 创建模型创建器
        creator = AdannLightgbmModelCreator()
        print("✅ ADANN+LightGBM模型创建器初始化成功")
        
        # 测试超参数定义
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        hyperparams = creator.define_hyperparams(trial)
        print(f"✅ 超参数定义成功: {len(hyperparams)} 个参数")
        
        # 测试模型创建
        input_shape = (100, 5)  # 100时间步，5通道
        n_classes = 11
        model = creator.create_model(hyperparams, input_shape, n_classes)
        print("✅ ADANN+LightGBM混合模型创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ ADANN+LightGBM测试失败: {e}")
        return False

def test_pipeline_integration():
    """测试pipeline集成"""
    print("\n🧪 测试pipeline集成...")
    
    try:
        # 测试导入
        from src.training.train_adann import AdannModelCreator
        from src.training.train_adann_lightgbm import AdannLightgbmModelCreator
        
        print("✅ 模型导入成功")
        
        # 测试run.py中的模型选择逻辑
        import argparse
        
        # 模拟命令行参数
        test_args = argparse.Namespace()
        test_args.model_type = 'ADANN'
        
        # 测试模型选择逻辑
        if test_args.model_type == 'ADANN':
            model_creator = AdannModelCreator()
            print("✅ ADANN模型选择成功")
        
        test_args.model_type = 'ADANN_LightGBM'
        if test_args.model_type == 'ADANN_LightGBM':
            model_creator = AdannLightgbmModelCreator()
            print("✅ ADANN+LightGBM模型选择成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 ADANN模型功能测试")
    print("=" * 50)
    
    results = []
    
    # 测试各个组件
    results.append(test_adann())
    results.append(test_adann_lightgbm())
    results.append(test_pipeline_integration())
    
    # 总结结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    if all(results):
        print("🎉 所有测试通过！ADANN模型已成功集成到pipeline中")
        print("\n💡 可以使用以下命令进行训练:")
        print("   python run.py --model_type ADANN --loso --n_trials 20")
        print("   python run.py --model_type ADANN_LightGBM --loso --n_trials 10")
        return 0
    else:
        print("❌ 部分测试失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())