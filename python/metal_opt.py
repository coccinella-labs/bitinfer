# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 harpertoken
import torch
import torch.nn as nn


class MetalOptimizedLinear(nn.Module):
    """FP16 linear layer optimized for Metal Performance Shaders"""

    def __init__(self, in_features, out_features, original_weight, original_bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store weights in FP16 for memory efficiency
        self.register_buffer("weight", original_weight.half())
        if original_bias is not None:
            self.register_buffer("bias", original_bias.half())
        else:
            self.bias = None

    def forward(self, x):
        # Ensure input is FP16 for optimal Metal performance
        if x.dtype != torch.float16:
            x = x.half()
        return torch.nn.functional.linear(x, self.weight, self.bias)


def metal_optimize_model(model):
    """Optimize model for Metal Performance Shaders"""
    # Convert entire model to FP16
    model = model.half()

    # Replace Linear layers with Metal-optimized versions
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            optimized_layer = MetalOptimizedLinear(
                module.in_features,
                module.out_features,
                module.weight.float(),  # Convert back to float32 temporarily
                module.bias.float() if module.bias is not None else None,
            )
            setattr(model, name, optimized_layer)
        else:
            # Recursively optimize child modules
            metal_optimize_model(module)

    return model
