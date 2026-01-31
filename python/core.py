# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 harpertoken
import torch
from transformers import AutoModel, AutoTokenizer

try:
    from .cache import ModelCache
    from .custom_quant import custom_quantize_model
    from .metal_opt import metal_optimize_model
    from .quantization import quantize_model
    from .streaming import StreamingInference
except ImportError:
    from cache import ModelCache
    from custom_quant import custom_quantize_model
    from metal_opt import metal_optimize_model
    from quantization import quantize_model
    from streaming import StreamingInference


class BitInfer:
    def __init__(
        self, model_name, device="mps", quantization="pytorch", use_cache=True
    ):
        self.device = device
        self.model_name = model_name
        self.quantization = quantization
        self.cache = ModelCache() if use_cache else None

        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Try to load from cache first
        cached_model = None
        if self.cache:
            cached_model = self.cache.load_model(self.model, model_name, quantization)

        if cached_model is not None:
            self.model = cached_model.to(device)
        else:
            # Apply optimization (PyTorch FP16 is now default - it's the fastest!)
            if quantization == "metal":
                print("[lightning] Using Metal-optimized FP16")
                self.model = metal_optimize_model(self.model).to(device)
            elif quantization == "custom":
                print("[wrench] Using custom 8-bit quantization")
                self.model = custom_quantize_model(self.model).to(device)
            else:
                print("[rocket] Using optimized PyTorch FP16 (fastest)")
                self.model = quantize_model(self.model).to(device)

            # Cache the optimized model
            if self.cache:
                self.cache.save_model(self.model, model_name, quantization)

        self.model.eval()

        # Initialize streaming interface
        self.streaming = StreamingInference(self)

    def infer(self, text):
        """Run inference on input text"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            # Convert inputs to FP16 for Metal optimization
            if self.quantization == "metal":
                inputs = {
                    k: v.half() if v.dtype == torch.float32 else v
                    for k, v in inputs.items()
                }
            outputs = self.model(**inputs)
            return outputs

    def batch_infer(self, texts):
        """Run batch inference"""
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(
                self.device
            )
            # Convert inputs to FP16 for Metal optimization
            if self.quantization == "metal":
                inputs = {
                    k: v.half() if v.dtype == torch.float32 else v
                    for k, v in inputs.items()
                }
            outputs = self.model(**inputs)
            return outputs

    def stream_infer(self, texts, batch_size=None):
        """Stream inference results"""
        return self.streaming.stream_infer(texts, batch_size)

    def adaptive_infer(self, texts, max_memory_mb=1000):
        """Adaptive batch inference based on memory"""
        return self.streaming.adaptive_batch_infer(texts, max_memory_mb)

    def clear_cache(self):
        """Clear model cache"""
        if self.cache:
            self.cache.clear_cache()

    def cache_info(self):
        """Get cache information"""
        if self.cache:
            return {
                "cache_size_mb": self.cache.cache_size(),
                "cache_dir": str(self.cache.cache_dir),
            }
        return None
