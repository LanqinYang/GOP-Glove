#!/usr/bin/env python3
"""
BSL手势识别训练调度器
简单的dispatcher，将训练任务分发给专门的训练脚本
"""

import sys
import subprocess
import os

def main():
    if len(sys.argv) < 2:
        print("使用方法: python train.py <model_type> [options]")
        print("支持的模型类型:")
        print("  1D_CNN")  
        print("  CNN_LSTM")
        print("  Transformer_Encoder")
        print("  XGBoost")
        print("")
        print("选项:")
        print("  --arduino     启用Arduino优化模式")
        print("  --epochs N    训练轮数")
        print("  --n_trials N  超参数优化试验次数")
        print("")
        print("智能早停:")
        print("  ✅ 准确率达到100%时自动停止")
        print("  ✅ 避免浪费计算时间")
        print("  📊 只在找到完美解时提前停止")
        print("")
        print("示例:")
        print("  python train.py 1D_CNN --arduino")
        print("  python train.py XGBoost --epochs 30 --n_trials 200")
        sys.exit(1)
    
    model_type = sys.argv[1]
    
    scripts = {
        "1D_CNN": "train_cnn1d.py",
        "CNN_LSTM": "train_cnn_lstm.py", 
        "Transformer_Encoder": "train_transformer.py",
        "XGBoost": "train_xgboost.py"
    }
    
    if model_type not in scripts:
        print(f"不支持的模型类型: {model_type}")
        print(f"支持的类型: {', '.join(scripts.keys())}")
        sys.exit(1)
    
    script_path = f"src/training/{scripts[model_type]}"
    
    # 检查脚本是否存在
    if not os.path.exists(script_path):
        print(f"错误: 训练脚本不存在: {script_path}")
        sys.exit(1)
    
    # 构建命令：添加--model_type参数，然后转发其他所有参数
    cmd = [sys.executable, script_path, '--model_type', model_type] + sys.argv[2:]
    
    print(f"🚀 启动 {model_type} 模型训练...")
    print(f"📝 执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    # 执行训练脚本
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 