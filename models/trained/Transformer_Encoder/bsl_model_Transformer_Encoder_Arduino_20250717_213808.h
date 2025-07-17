/*
 * BSL Gesture Recognition Model
 *
 * Model Type: Transformer_Encoder_Arduino
 * Timestamp:  20250717_213808
 * Model Size: 24 bytes
 */

#ifndef BSL_MODEL_H_20250717_213808
#define BSL_MODEL_H_20250717_213808

// Feature count for normalization
const int BSL_MODEL_FEATURES = 5;

// Normalization parameters (StandardScaler)
const float scaler_mean[BSL_MODEL_FEATURES] = { 236.43887165f, 152.37936569f, 328.61752741f, 316.69635454f, 101.83935373f };
const float scaler_scale[BSL_MODEL_FEATURES] = { 177.03868095f, 153.40072310f, 204.37148658f, 203.09171501f, 121.78085463f };

// TFLite model data
alignas(16) const unsigned char model_data[] = {
    0x54, 0x46, 0x4c, 0x49, 0x54, 0x45, 0x5f, 0x43, 0x4f, 0x4e, 0x56, 0x45, 0x52, 0x53, 0x49, 0x4f, 0x4e, 0x5f, 0x46, 0x41, 0x49, 0x4c, 0x45, 0x44
};
const unsigned int model_data_len = 24;

#endif // BSL_MODEL_H_20250717_213808
