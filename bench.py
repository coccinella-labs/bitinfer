"""Reproduces the performance table in README.md.

Run: python bench.py

The README previously advertised speedups that this script cannot reproduce.
The memory reduction is real and consistent. The speedup is not: the ratio
moves with machine load by more than any effect being measured, so the honest
result is parity.

Both sides are timed in the same process, after a warmup call, taking the
median of N timed calls, and the whole comparison is repeated so that a single
noisy run cannot be mistaken for a result.
"""

import statistics
import sys
import time

import torch

sys.path.insert(0, "python")

from core import BitInfer  # noqa: E402
from quantization import get_model_size  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

TEXT = "Benchmark test for inference timing"
CALLS = 40
REPEATS = 3

MODELS = [
    "hf-internal-testing/tiny-random-bert",
    "google/bert_uncased_L-2_H-128_A-2",
]


def median_ms(fn, calls=CALLS):
    fn()  # warm up
    timings = []
    for _ in range(calls):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def compare(model_name):
    """Return (bitinfer_ms, transformers_ms) for one model."""
    bitinfer = BitInfer(model_name, device="cpu", use_cache=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baseline = AutoModel.from_pretrained(model_name).eval()

    def run_baseline():
        with torch.no_grad():
            return baseline(**tokenizer(TEXT, return_tensors="pt"))

    bitinfer_ms = []
    baseline_ms = []
    for _ in range(REPEATS):
        baseline_ms.append(median_ms(run_baseline))
        bitinfer_ms.append(median_ms(lambda: bitinfer.infer(TEXT)))

    return (
        statistics.median(bitinfer_ms),
        statistics.median(baseline_ms),
        get_model_size(baseline),
        get_model_size(bitinfer.model),
    )


def main():
    print("| model | BitInfer | transformers | ratio | memory MB |")
    print("|---|---|---|---|---|")

    for model_name in MODELS:
        try:
            ours, theirs, mem_theirs, mem_ours = compare(model_name)
        except Exception as exc:  # noqa: BLE001
            print(f"| {model_name} | skipped: {type(exc).__name__}: {exc} |")
            continue

        label = model_name.split("/")[-1]
        ratio = theirs / ours
        print(
            f"| {label} | {ours:.2f}ms | {theirs:.2f}ms | "
            f"{ratio:.2f}x | {mem_theirs:.1f} -> {mem_ours:.1f} |"
        )

    print(
        "\nRatio is transformers/BitInfer, so above 1.00x means BitInfer is "
        "faster.\nMemory is the float32 -> float16 change and is the reliable "
        "result here."
    )


if __name__ == "__main__":
    main()
