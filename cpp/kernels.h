#ifndef BITINFER_KERNELS_H
#define BITINFER_KERNELS_H

#include <vector>

// Fast matrix multiplication for quantized weights
void quantized_matmul(
    const float* input,
    const int8_t* weight,
    float* output,
    int batch_size,
    int input_dim,
    int output_dim
);

// Batch quantization
void batch_quantize(
    const float* input,
    int8_t* output,
    float* scale,
    int size
);

#endif
