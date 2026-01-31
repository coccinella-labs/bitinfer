# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 harpertoken
import numpy as np
import torch
import torch.nn as nn


class OptimizedQuantizedLinear(nn.Module):
    """Optimized 8-bit quantized linear layer for M1"""

    def __init__(self, in_features, out_features, original_weight, original_bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Pre-compute quantized weights and store as int8
        weight_np = original_weight.detach().cpu().numpy()

        # Per-channel quantization for better accuracy
        weight_scales = []
        weight_zero_points = []
        quantized_weights = []

        for i in range(out_features):
            channel_weight = weight_np[i, :]
            w_min, w_max = channel_weight.min(), channel_weight.max()
            scale = (w_max - w_min) / 255.0 if w_max != w_min else 1.0
            zero_point = w_min

            quantized = ((channel_weight - zero_point) / scale).astype(np.int8)

            weight_scales.append(scale)
            weight_zero_points.append(zero_point)
            quantized_weights.append(quantized)

        # Store as tensors
        self.register_buffer(
            "weight_quantized",
            torch.from_numpy(np.array(quantized_weights, dtype=np.int8)),
        )
        self.register_buffer(
            "weight_scales", torch.tensor(weight_scales, dtype=torch.float32)
        )
        self.register_buffer(
            "weight_zero_points", torch.tensor(weight_zero_points, dtype=torch.float32)
        )

        if original_bias is not None:
            self.register_buffer("bias", original_bias)
        else:
            self.bias = None

    def forward(self, x):
        # Fast vectorized dequantization
        weight_fp = self.weight_quantized.float() * self.weight_scales.unsqueeze(
            1
        ) + self.weight_zero_points.unsqueeze(1)
        return torch.nn.functional.linear(x, weight_fp, self.bias)


def custom_quantize_model(model):
    """Replace Linear layers with optimized quantized versions"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace with optimized quantized layer
            quantized_layer = OptimizedQuantizedLinear(
                module.in_features, module.out_features, module.weight, module.bias
            )
            setattr(model, name, quantized_layer)
        else:
            # Recursively quantize child modules
            custom_quantize_model(module)

    return model
