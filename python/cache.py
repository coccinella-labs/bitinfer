# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 coccinella-labs
import hashlib
import os  # noqa: F401
import pickle
from pathlib import Path

import torch


class ModelCache:
    """Cache optimized models to disk for faster loading"""

    def __init__(self, cache_dir="~/.bitinfer_cache"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, model_name, quantization):
        """Generate cache key from model name and quantization type"""
        key_str = f"{model_name}_{quantization}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key):
        """Get full path to cached model"""
        return self.cache_dir / f"{cache_key}.pkl"

    def save_model(self, model, model_name, quantization):
        """Save optimized model to cache"""
        cache_key = self._get_cache_key(model_name, quantization)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(model.state_dict(), f)
            print(f"[floppy_disk] Cached model: {cache_path}")
            return True
        except Exception as e:
            print(f"[warning]  Cache save failed: {e}")
            return False

    def load_model(self, model_template, model_name, quantization):
        """Load optimized model from cache.

        `load_state_dict` copies each tensor into the destination parameter's
        own dtype, and the template comes from `from_pretrained` in float32, so
        a float16 model cached as float16 was silently upcast to float32 on the
        next run. The "optimized" model then ran unoptimized while reporting a
        cache hit. The model is therefore cast back to the cached dtype after
        loading, which is the whole point of caching it.
        """
        cache_key = self._get_cache_key(model_name, quantization)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                state_dict = pickle.load(f)

            cached_dtype = next(
                (v.dtype for v in state_dict.values() if torch.is_tensor(v)), None
            )
            if cached_dtype is None:
                return None

            model_template.load_state_dict(state_dict)
            model_template = model_template.to(cached_dtype)
            print(
                f"[lightning] Loaded from cache: {cache_path} "
                f"({cached_dtype})"
            )
            return model_template
        except Exception as e:
            print(f"[warning]  Cache load failed: {e}")
            return None

    def clear_cache(self):
        """Clear all cached models"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("[trash]  Cache cleared")

    def cache_size(self):
        """Get total cache size in MB"""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        return total_size / 1024 / 1024
