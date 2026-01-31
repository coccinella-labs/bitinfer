#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 harpertoken

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bitinfer import BitInfer


def main():
    print("BitInfer Basic Usage Example")

    # Initialize with a small model
    print("Loading model...")
    model = BitInfer("distilbert-base-uncased")

    # Single inference
    print("Running inference...")
    result = model.infer("This is a test sentence.")
    print(f"Output shape: {result.last_hidden_state.shape}")

    # Batch inference
    print("Running batch inference...")
    texts = ["Hello world", "How are you?", "BitInfer is fast!"]
    batch_result = model.batch_infer(texts)
    print(f"Batch output shape: {batch_result.last_hidden_state.shape}")

    print("Done!")


if __name__ == "__main__":
    main()
