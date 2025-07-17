/*
 * BSL Gesture Recognition Model
 *
 * Model Type: 1D_CNN_Arduino
 * Timestamp:  20250717_212355
 * Model Size: 26 bytes
 */

#ifndef BSL_MODEL_H_20250717_212355
#define BSL_MODEL_H_20250717_212355

// Feature count for normalization
const int BSL_MODEL_FEATURES = 5;

// Normalization parameters (StandardScaler)
const float scaler_mean[BSL_MODEL_FEATURES] = { 236.43887165f, 152.37936569f, 328.61752741f, 316.69635454f, 101.83935373f };
const float scaler_scale[BSL_MODEL_FEATURES] = { 177.03868095f, 153.40072310f, 204.37148658f, 203.09171501f, 121.78085463f };

// TFLite model data
alignas(16) const unsigned char model_data[] = {
    0x54, 0x46, 0x4c, 0x49, 0x54, 0x45, 0x5f, 0x51, 0x55, 0x41, 0x4e, 0x54, 0x49, 0x5a, 0x41, 0x54, 0x49, 0x4f, 0x4e, 0x5f, 0x46, 0x41, 0x49, 0x4c, 0x45, 0x44
};
const unsigned int model_data_len = 26;

#endif // BSL_MODEL_H_20250717_212355
