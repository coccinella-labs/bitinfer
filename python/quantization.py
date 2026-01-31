# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 harpertoken
import torch
import torch.nn as nn


def quantize_model(model, dtype=torch.qint8):
    """Quantize model for efficient inference - M1 compatible version"""
    try:
        # Try PyTorch's dynamic quantization first
        quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv1d, nn.Conv2d}, dtype=dtype
        )
        return quantized
    except RuntimeError as e:
        if "NoQEngine" in str(e):  # noqa: E501
            print(
                "[warning]  PyTorch quantization not available on M1, using FP16 instead"
            )
            # Fallback to half precision for M1
            return model.half()
        else:
            raise e


def get_model_size(model):
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024
