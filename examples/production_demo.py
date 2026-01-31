#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 harpertoken

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from core import BitInfer


def demo_production_features():
    print("[rocket] BitInfer Production Features Demo")
    print("=" * 50)

    # Initialize BitInfer with caching
    print("[package] Loading model with caching enabled...")
    model = BitInfer("prajjwal1/bert-tiny", use_cache=True)

    # Cache info
    cache_info = model.cache_info()
    print(
        f"[floppy_disk] Cache info: {cache_info['cache_size_mb']:.2f} MB in {cache_info['cache_dir']}"
    )

    # Single inference
    print("\n🔍 Single inference:")
    result = model.infer("BitInfer is fast and efficient!")
    print(f"[check] Output shape: {result.last_hidden_state.shape}")

    # Batch inference
    print("\n🔍 Batch inference:")
    texts = [
        "Hello world",
        "How are you?",
        "BitInfer rocks!",
        "Machine learning is cool",
    ]
    batch_result = model.batch_infer(texts)
    print(f"[check] Batch output shape: {batch_result.last_hidden_state.shape}")

    # Streaming inference
    print("\n[ocean] Streaming inference:")
    stream_texts = [f"Stream text {i}" for i in range(10)]

    start_time = time.time()
    for i, result in enumerate(model.stream_infer(stream_texts, batch_size=3)):
        print(
            f"  Stream {result['batch_index']}: {result['text'][:20]}... -> {result['embedding'].shape}"
        )
        if i >= 4:  # Show first 5 results
            print("  ...")
            break
    stream_time = time.time() - start_time
    print(f"[lightning] Streaming completed in {stream_time:.3f}s")

    # Adaptive inference
    print("\n[brain] Adaptive inference (memory-aware):")
    adaptive_texts = [
        f"Adaptive text {i} with varying lengths and complexity" for i in range(20)
    ]

    start_time = time.time()
    results_count = 0
    for result in model.adaptive_infer(adaptive_texts, max_memory_mb=500):
        results_count += 1
        if results_count <= 3:
            print(f"  Batch {results_count}: {result['embedding'].shape}")
    adaptive_time = time.time() - start_time
    print(
        f"[lightning] Adaptive inference: {results_count} results in {adaptive_time:.3f}s"
    )

    # Performance comparison
    print("\n[chart] Performance comparison:")
    test_texts = ["Performance test"] * 50

    # Regular batch
    start_time = time.time()
    model.batch_infer(test_texts)
    batch_time = time.time() - start_time

    # Streaming
    start_time = time.time()
    list(model.stream_infer(test_texts, batch_size=10))
    streaming_time = time.time() - start_time

    print(f"  Batch inference: {batch_time:.3f}s")
    print(f"  Streaming inference: {streaming_time:.3f}s")
    print(
        f"  Streaming overhead: {((streaming_time - batch_time) / batch_time * 100):.1f}%"
    )

    # Cache management
    print(
        f"\n[floppy_disk] Final cache size: {model.cache_info()['cache_size_mb']:.2f} MB"
    )

    print("\n[party] Production features demo complete!")


if __name__ == "__main__":
    demo_production_features()
