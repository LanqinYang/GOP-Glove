#!/usr/bin/env python3
"""
快速模型测试脚本
验证所有BSL手势识别模型是否可以正常加载和推理
"""

import os
import time
import numpy as np
import torch
import tensorflow as tf
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def test_model_loading():
    """测试模型加载"""
    print("🔍 测试模型加载...")
    
    # 定义要测试的模型
    models_to_test = [
        ("1D_CNN", "models/trained/1D_CNN/standard/full"),
        ("Transformer_Encoder", "models/trained/Transformer_Encoder/standard/full"),
        ("LightGBM", "models/trained/LightGBM/standard/full"),
        ("XGBoost", "models/trained/XGBoost/standard/full"),
        ("ADANN_LightGBM", "models/trained/ADANN_LightGBM/standard/full")
    ]
    
    results = {}
    
    for model_type, model_dir in models_to_test:
        print(f"\n📁 测试 {model_type}...")
        
        if not os.path.exists(model_dir):
            print(f"❌ 目录不存在: {model_dir}")
            continue
        
        # 查找模型文件
        files = os.listdir(model_dir)
        model_files = []
        
        if model_type in ["1D_CNN", "Transformer_Encoder"]:
            for file in files:
                if file.endswith('.keras') or file.endswith('.tflite'):
                    model_files.append(file)
        elif model_type in ["LightGBM", "XGBoost"]:
            for file in files:
                if file.endswith('.pkl') and not file.startswith('scaler_'):
                    model_files.append(file)
        elif model_type == "ADANN_LightGBM":
            for file in files:
                if file.endswith('.pth'):
                    model_files.append(file)
        
        if not model_files:
            print(f"❌ 未找到 {model_type} 模型文件")
            continue
        
        # 选择最新的模型文件
        model_files.sort(reverse=True)
        latest_model = model_files[0]
        model_path = os.path.join(model_dir, latest_model)
        
        print(f"✅ 找到模型: {latest_model}")
        
        # 尝试加载模型
        try:
            if model_type == "1D_CNN":
                # 尝试Keras格式
                try:
                    model = tf.keras.models.load_model(model_path)
                    print(f"✅ 成功加载Keras模型")
                    results[model_type] = {"status": "success", "framework": "keras", "path": model_path}
                except:
                    # 尝试TFLite格式
                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()
                    print(f"✅ 成功加载TFLite模型")
                    results[model_type] = {"status": "success", "framework": "tflite", "path": model_path}
                    
            elif model_type == "Transformer_Encoder":
                try:
                    model = tf.keras.models.load_model(model_path)
                    print(f"✅ 成功加载Keras模型")
                    results[model_type] = {"status": "success", "framework": "keras", "path": model_path}
                except:
                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()
                    print(f"✅ 成功加载TFLite模型")
                    results[model_type] = {"status": "success", "framework": "tflite", "path": model_path}
                    
            elif model_type == "LightGBM":
                model = lgb.Booster(model_file=model_path)
                print(f"✅ 成功加载LightGBM模型")
                results[model_type] = {"status": "success", "framework": "lightgbm", "path": model_path}
                
            elif model_type == "XGBoost":
                model = xgb.Booster()
                model.load_model(model_path)
                print(f"✅ 成功加载XGBoost模型")
                results[model_type] = {"status": "success", "framework": "xgboost", "path": model_path}
                
            elif model_type == "ADANN_LightGBM":
                # 导入必要的模块
                import sys
                if 'src' not in sys.path:
                    sys.path.append('src')
                if 'src/training' not in sys.path:
                    sys.path.append('src/training')

                from train_adann_lightgbm import AdannLightgbmModelCreator, AdannLightgbmModelWrapper

                # 尝试加载新版保存包（包含完整对象）或兼容旧版
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                package = torch.load(model_path, map_location=device)

                model_creator = AdannLightgbmModelCreator()
                # 统一封装为 wrapper，便于后续推理
                params = package.get('params', {}) if isinstance(package, dict) else {}
                wrapper = model_creator.create_model(params)

                # 新版：直接解包
                if isinstance(package, dict) and 'adann_model' in package:
                    wrapper.hybrid_model['adann'] = package['adann_model']
                    wrapper.hybrid_model['lightgbm'] = package['lightgbm_model']
                    wrapper.hybrid_model['ensemble_weight'] = package.get('ensemble_weight', 0.5)
                    # 附加工具对象
                    for key in ['gesture_encoder', 'subject_encoder', 'adann_scaler', 'lgb_scaler', 'hybrid_extractor']:
                        if key in package:
                            wrapper.hybrid_model[key] = package[key]
                    wrapper.trained = True
                # 旧版：仅 state_dict
                elif isinstance(package, dict) and 'adann_state_dict' in package:
                    reconstructed = model_creator.create_model(params)
                    reconstructed.hybrid_model['adann'].load_state_dict(package['adann_state_dict'])
                    reconstructed.hybrid_model['lightgbm'] = package['lightgbm_model']
                    reconstructed.hybrid_model['ensemble_weight'] = package.get('ensemble_weight', 0.5)
                    wrapper = reconstructed
                    wrapper.trained = True
                else:
                    raise ValueError("Unrecognized ADANN_LightGBM package format")

                print(f"✅ 成功加载ADANN_LightGBM模型")
                results[model_type] = {"status": "success", "framework": "adann_lightgbm", "path": model_path}
                
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            results[model_type] = {"status": "failed", "error": str(e), "path": model_path}
    
    return results

def test_inference():
    """测试推理功能"""
    print("\n🚀 测试推理功能...")
    
    # 生成测试数据 - 根据模型调整格式
    test_data_3d = np.random.randn(1, 6, 100).astype(np.float32)  # 原始3D格式
    test_data_tflite = np.random.randn(1, 100, 5).astype(np.float32)  # TFLite格式
    test_data_2d = np.random.randn(1, 600).astype(np.float32)      # 2D格式
    print(f"📊 测试数据形状: 3D={test_data_3d.shape}, TFLite={test_data_tflite.shape}, 2D={test_data_2d.shape}")
    
    # 测试每种模型类型
    model_results = test_model_loading()
    
    for model_type, result in model_results.items():
        if result["status"] != "success":
            print(f"⚠️ 跳过 {model_type} (加载失败)")
            continue
            
        print(f"\n🧪 测试 {model_type} 推理...")
        
        try:
            if result["framework"] == "keras":
                model = tf.keras.models.load_model(result["path"])
                start_time = time.time()
                _ = model.predict(test_data_3d, verbose=0)
                latency = (time.time() - start_time) * 1000
                print(f"✅ Keras推理成功: {latency:.2f}ms")
                
            elif result["framework"] == "tflite":
                interpreter = tf.lite.Interpreter(model_path=result["path"])
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # 检查输入格式并调整
                input_shape = input_details[0]['shape']
                print(f"📊 TFLite输入形状: {input_shape}")
                
                if list(input_shape) == [1, 100, 5]:
                    test_data = test_data_tflite
                elif len(input_shape) == 3:
                    test_data = test_data_3d
                else:
                    test_data = test_data_2d
                
                interpreter.set_tensor(input_details[0]['index'], test_data)
                start_time = time.time()
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
                latency = (time.time() - start_time) * 1000
                print(f"✅ TFLite推理成功: {latency:.2f}ms")
                
            elif result["framework"] == "lightgbm":
                model = lgb.Booster(model_file=result["path"])
                start_time = time.time()
                _ = model.predict(test_data_2d)
                latency = (time.time() - start_time) * 1000
                print(f"✅ LightGBM推理成功: {latency:.2f}ms")
                
            elif result["framework"] == "xgboost":
                model = xgb.Booster()
                model.load_model(result["path"])
                dmatrix = xgb.DMatrix(test_data_2d)
                start_time = time.time()
                _ = model.predict(dmatrix)
                latency = (time.time() - start_time) * 1000
                print(f"✅ XGBoost推理成功: {latency:.2f}ms")
                
            elif result["framework"] == "adann_lightgbm":
                import sys
                if 'src/training' not in sys.path:
                    sys.path.append('src/training')
                from train_adann_lightgbm import AdannLightgbmModelCreator, AdannLightgbmModelWrapper

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                package = torch.load(result["path"], map_location=device)

                model_creator = AdannLightgbmModelCreator()
                params = package.get('params', {}) if isinstance(package, dict) else {}
                wrapper = model_creator.create_model(params)

                # 新旧格式兼容
                if isinstance(package, dict) and 'adann_model' in package:
                    wrapper.hybrid_model['adann'] = package['adann_model']
                    wrapper.hybrid_model['lightgbm'] = package['lightgbm_model']
                    wrapper.hybrid_model['ensemble_weight'] = package.get('ensemble_weight', 0.5)
                    for key in ['gesture_encoder', 'subject_encoder', 'adann_scaler', 'lgb_scaler', 'hybrid_extractor']:
                        if key in package:
                            wrapper.hybrid_model[key] = package[key]
                elif isinstance(package, dict) and 'adann_state_dict' in package:
                    wrapper.hybrid_model['adann'].load_state_dict(package['adann_state_dict'])
                    wrapper.hybrid_model['lightgbm'] = package['lightgbm_model']
                    wrapper.hybrid_model['ensemble_weight'] = package.get('ensemble_weight', 0.5)
                else:
                    raise ValueError("Unrecognized ADANN_LightGBM package format")

                # 调用概率输出接口以避免类别-概率不一致
                # ADANN/Hybrid 期望输入 (batch, 100, 5)
                test_data_for_hybrid = test_data_tflite
                start_time = time.time()
                _ = model_creator.predict_hybrid(wrapper.hybrid_model, test_data_for_hybrid)
                latency = (time.time() - start_time) * 1000
                print(f"✅ ADANN_LightGBM推理成功: {latency:.2f}ms")
                
        except Exception as e:
            print(f"❌ {model_type} 推理失败: {e}")

def main():
    """主函数"""
    print("🔬 BSL手势识别模型快速测试")
    print("=" * 50)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU可用: {torch.cuda.get_device_name()}")
    else:
        print("💻 使用CPU")
    
    # 测试模型加载
    model_results = test_model_loading()
    
    # 打印汇总
    print("\n📋 模型加载汇总:")
    for model_type, result in model_results.items():
        if result["status"] == "success":
            print(f"✅ {model_type}: {result['framework']}")
        else:
            print(f"❌ {model_type}: 加载失败")
    
    # 测试推理
    test_inference()
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    main()
