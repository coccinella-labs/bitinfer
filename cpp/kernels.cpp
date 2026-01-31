#include "kernels.h"
#include <cmath>
#include <algorithm>

void quantized_matmul(
    const float* input,
    const int8_t* weight,
    float* output,
    int batch_size,
    int input_dim,
    int output_dim
) {
    for (int b = 0; b < batch_size; ++b) {
        for (int o = 0; o < output_dim; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; ++i) {
                sum += input[b * input_dim + i] * static_cast<float>(weight[o * input_dim + i]);
            }
            output[b * output_dim + o] = sum;
        }
    }
}

void batch_quantize(
    const float* input,
    int8_t* output,
    float* scale,
    int size
) {
    // Find min/max
    float min_val = input[0], max_val = input[0];
    for (int i = 1; i < size; ++i) {
        min_val = std::min(min_val, input[i]);
        max_val = std::max(max_val, input[i]);
    }

    // Calculate scale
    *scale = (max_val - min_val) / 255.0f;

    // Quantize
    for (int i = 0; i < size; ++i) {
        output[i] = static_cast<int8_t>((input[i] - min_val) / (*scale) - 128);
    }
}
