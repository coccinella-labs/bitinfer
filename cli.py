#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 harpertoken

import argparse
import json
import os
import sys

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from core import BitInfer


def main():
    parser = argparse.ArgumentParser(
        description="BitInfer - Fast inference framework for Apple Silicon"
    )

    # Model selection
    parser.add_argument(
        "model", help="Hugging Face model name (e.g., distilbert-base-uncased)"
    )

    # Input options
    parser.add_argument("--text", "-t", help="Single text to process")
    parser.add_argument("--file", "-f", help="File containing texts (one per line)")
    parser.add_argument("--batch", "-b", nargs="+", help="Multiple texts as arguments")

    # Optimization options
    parser.add_argument(
        "--quantization",
        "-q",
        choices=["pytorch", "metal", "custom"],
        default="pytorch",
        help="Quantization method (default: pytorch)",
    )
    parser.add_argument(
        "--device", "-d", default="mps", help="Device to use (default: mps)"
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable model caching")

    # Output options
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument(
        "--format", choices=["json", "text"], default="text", help="Output format"
    )
    parser.add_argument(
        "--streaming", "-s", action="store_true", help="Use streaming inference"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for streaming"
    )

    # Utility commands
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--cache-info", action="store_true", help="Show cache information"
    )
    parser.add_argument("--clear-cache", action="store_true", help="Clear model cache")

    args = parser.parse_args()

    # Handle utility commands
    if args.clear_cache:
        model = BitInfer(args.model, use_cache=True)
        model.clear_cache()
        print("[trash]  Cache cleared")
        return

    if args.cache_info:
        model = BitInfer(args.model, use_cache=True)
        info = model.cache_info()
        if info:
            print(f"[floppy_disk] Cache size: {info['cache_size_mb']:.2f} MB")
            print(f"[folder] Cache directory: {info['cache_dir']}")
        else:
            print("[x] Caching disabled")
        return

    # Initialize model
    print(f"[rocket] Loading {args.model}...")
    model = BitInfer(
        args.model,
        device=args.device,
        quantization=args.quantization,
        use_cache=not args.no_cache,
    )

    # Collect texts to process
    texts = []
    if args.text:
        texts.append(args.text)
    elif args.file:
        with open(args.file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.batch:
        texts = args.batch
    else:
        print("[x] No input provided. Use --text, --file, or --batch")
        return

    print(f"[memo] Processing {len(texts)} text(s)...")

    # Run benchmark if requested
    if args.benchmark:
        import time

        start_time = time.time()

        if args.streaming:
            results = list(model.stream_infer(texts, batch_size=args.batch_size))
        else:
            if len(texts) == 1:
                results = [model.infer(texts[0])]
            else:
                results = [model.batch_infer(texts)]

        end_time = time.time()

        print(f"[lightning] Processed in {end_time - start_time:.3f}s")
        print(
            f"[chart] Average: {(end_time - start_time) / len(texts) * 1000:.1f}ms per text"
        )
        return

    # Process texts
    if args.streaming and len(texts) > 1:
        results = []
        for result in model.stream_infer(texts, batch_size=args.batch_size):
            results.append(
                {
                    "text": result["text"],
                    "embedding_shape": list(result["embedding"].shape),
                    "batch_index": result["batch_index"],
                }
            )
    else:
        if len(texts) == 1:
            output = model.infer(texts[0])
            results = [
                {
                    "text": texts[0],
                    "embedding_shape": list(output.last_hidden_state.shape),
                }
            ]
        else:
            output = model.batch_infer(texts)
            results = [
                {
                    "texts": texts,
                    "embedding_shape": list(output.last_hidden_state.shape),
                }
            ]

    # Output results
    if args.format == "json":
        output_data = json.dumps(results, indent=2)
    else:
        output_lines = []
        for i, result in enumerate(results):
            if "texts" in result:
                output_lines.append(f"Batch result: {result['embedding_shape']}")
            else:
                output_lines.append(f"Text {i+1}: {result['text'][:50]}...")
                output_lines.append(f"  Shape: {result['embedding_shape']}")
        output_data = "\n".join(output_lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_data)
        print(f"[floppy_disk] Results saved to {args.output}")
    else:
        print("\n[clipboard] Results:")
        print(output_data)


if __name__ == "__main__":
    main()
