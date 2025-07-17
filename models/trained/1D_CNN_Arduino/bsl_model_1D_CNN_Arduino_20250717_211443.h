/*
 * BSL手势识别模型
 * 模型类型: 1D_CNN_Arduino
 * 时间戳: 20250717_211443
 * 模型大小: 4 字节
 */

#ifndef BSL_MODEL_H_20250717211443
#define BSL_MODEL_H_20250717211443

// 特征数量
const int BSL_MODEL_FEATURES = 5;

// 归一化参数
const float scaler_mean[BSL_MODEL_FEATURES] = { 236.43887165f, 152.37936569f, 328.61752741f, 316.69635454f, 101.83935373f };
const float scaler_scale[BSL_MODEL_FEATURES] = { 177.03868095f, 153.40072310f, 204.37148658f, 203.09171501f, 121.78085463f };

// TFLite模型数据
alignas(16) const unsigned char model_data[] = {
    0x00, 0x00, 0x00, 0x00  // TFLite转换失败占位符
};
const unsigned int model_data_len = 4;

#endif
