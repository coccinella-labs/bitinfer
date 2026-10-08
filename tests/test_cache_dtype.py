# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 coccinella-labs
"""Regression test for the cache dtype bug.

Before the fix, a model cached as float16 came back as float32 on the next run:
`load_state_dict` copies into the destination parameter's dtype, and the
template comes from `from_pretrained` in float32. The cache reported a hit while
silently discarding the optimization that was the reason for caching.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch  # noqa: E402

from cache import ModelCache  # noqa: E402

MODEL = "hf-internal-testing/tiny-random-bert"


def _template():
    from transformers import AutoModel

    return AutoModel.from_pretrained(MODEL)


def test_cache_preserves_dtype(tmp_path):
    """A cached float16 model must come back as float16, not float32."""
    cache = ModelCache(cache_dir=tmp_path)

    source = _template()
    cache.save_model(source.half(), MODEL, "pytorch")

    reloaded = _template()
    assert next(reloaded.parameters()).dtype == torch.float32

    restored = cache.load_model(reloaded, MODEL, "pytorch")
    assert restored is not None, "cache should report a hit"
    assert next(restored.parameters()).dtype == torch.float16, (
        "cache returned float32: the cached float16 weights were upcast by "
        "load_state_dict, which silently discarded the optimization"
    )


def test_cache_miss_returns_none(tmp_path):
    cache = ModelCache(cache_dir=tmp_path)
    assert cache.load_model(_template(), "never-cached-model", "pytorch") is None


def test_saved_cache_is_reported_by_cache_size(tmp_path):
    """cache_size() must reflect a saved model.

    ModelCache has no cache_info(); BitInfer.cache_info() builds the dict from
    cache_size() and cache_dir, so this covers the same contract from below.
    """
    cache = ModelCache(cache_dir=tmp_path)
    assert cache.cache_size() == 0.0

    cache.save_model(_template().half(), MODEL, "pytorch")
    assert cache.cache_size() > 0.0


if __name__ == "__main__":
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as _d:
        test_cache_preserves_dtype(_d)
        test_cache_miss_returns_none(_d)
        test_saved_cache_is_reported_by_cache_size(_d)
    print("cache dtype tests passed")
