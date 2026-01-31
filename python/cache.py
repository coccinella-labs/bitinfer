# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 harpertoken
import hashlib
import os  # noqa: F401
import pickle
from pathlib import Path


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
        """Load optimized model from cache"""
        cache_key = self._get_cache_key(model_name, quantization)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                state_dict = pickle.load(f)
            model_template.load_state_dict(state_dict)
            print(f"[lightning] Loaded from cache: {cache_path}")
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
