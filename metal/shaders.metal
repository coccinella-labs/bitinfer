#include <metal_stdlib>
using namespace metal;

// Fast quantized matrix multiplication on GPU
kernel void quantized_matmul_metal(
    device const float* input [[buffer(0)]],
    device const char* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint output_idx = gid.x;

    if (output_idx >= output_dim) return;

    float sum = 0.0f;
    for (uint i = 0; i < input_dim; ++i) {
        sum += input[batch_idx * input_dim + i] * float(weight[output_idx * input_dim + i]);
    }

    output[batch_idx * output_dim + output_idx] = sum;
}

// Parallel quantization kernel
kernel void batch_quantize_metal(
    device const float* input [[buffer(0)]],
    device char* output [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    // Simple per-element quantization (scale computed on CPU)
    output[gid] = char((input[gid] / scale[0]) - 128);
}
