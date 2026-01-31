# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 harpertoken
from typing import Iterator, List, Union

import torch


class StreamingInference:
    """Streaming inference for real-time applications"""

    def __init__(self, bitinfer_model):
        self.model = bitinfer_model
        self.batch_buffer = []
        self.batch_size = 4  # Optimal batch size for M1

    def stream_infer(
        self, texts: Union[str, List[str]], batch_size: int = None
    ) -> Iterator:
        """Stream inference results as they become available"""
        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.batch_size

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Run batch inference
            with torch.no_grad():
                inputs = self.model.tokenizer(
                    batch, return_tensors="pt", padding=True
                ).to(self.model.device)
                if self.model.quantization == "metal":
                    inputs = {
                        k: v.half() if v.dtype == torch.float32 else v
                        for k, v in inputs.items()
                    }
                outputs = self.model.model(**inputs)

            # Yield individual results
            for j, text in enumerate(batch):
                yield {
                    "text": text,
                    "embedding": outputs.last_hidden_state[j],
                    "batch_index": i + j,
                }

    def batch_stream_infer(self, text_batches: List[List[str]]) -> Iterator:
        """Stream inference for pre-batched data"""
        for batch_idx, batch in enumerate(text_batches):
            results = self.model.batch_infer(batch)

            yield {
                "batch_index": batch_idx,
                "batch_size": len(batch),
                "results": results,
                "texts": batch,
            }

    def adaptive_batch_infer(
        self, texts: List[str], max_memory_mb: float = 1000
    ) -> Iterator:
        """Adaptive batching based on available memory"""
        # Estimate memory usage per text (rough approximation)
        avg_tokens = sum(
            len(self.model.tokenizer.encode(text)) for text in texts[:10]
        ) / min(10, len(texts))
        estimated_memory_per_text = (
            avg_tokens * 4 * 2
        )  # 4 bytes per float, 2 for forward pass

        # Calculate optimal batch size
        optimal_batch_size = max(
            1, int((max_memory_mb * 1024 * 1024) / estimated_memory_per_text)
        )
        optimal_batch_size = min(optimal_batch_size, 32)  # Cap at 32

        print(f"[brain] Adaptive batching: {optimal_batch_size} texts per batch")

        # Stream with optimal batch size
        yield from self.stream_infer(texts, batch_size=optimal_batch_size)
