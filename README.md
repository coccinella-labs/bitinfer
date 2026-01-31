# BitInfer

<!-- Toggle between styles: Click to switch -->
<details>
<summary>[wave] Personal Style (Click to expand)</summary>

## Hey, meet BitInfer.

BitInfer makes AI models run faster on your M1 Mac -- usually 1.6x faster, sometimes more. We built it because waiting for inference results is annoying, and 8GB of RAM shouldn't limit what you can do.

We've tested it with models from Hugging Face, optimized it for Apple Silicon, and made it dead simple to use. The performance gains are nice, but it's the "wow, that actually worked" moments when you see your model fly that we remember most.

### What it does

```python
from bitinfer import BitInfer

# Your model, but faster
model = BitInfer("distilbert-base-uncased")
result = model.infer("Hello world")  # 1.6x faster than vanilla PyTorch
```

BitInfer automatically:
- Cuts memory usage in half (FP16 optimization)
- Caches optimized models for instant loading
- Streams results for real-time apps
- Adapts batch sizes to your available memory

### Why we built this

M1 Macs are incredible machines, but most AI frameworks treat them like afterthoughts. We wanted something that actually leveraged Metal Performance Shaders properly and didn't require you to think about quantization schemes or memory management.

After building and benchmarking, we had something that consistently beats vanilla PyTorch while using half the memory.

### Getting started

```bash
# Clone and run
git clone https://github.com/harpertoken/bitinfer
cd bitinfer
pip install -r requirements.txt

# Try it out
python cli.py prajjwal1/bert-tiny --text "Hello BitInfer!" --benchmark
```

### The numbers

- **BERT-tiny**: 4.8ms vs 7.9ms (1.65x faster)
- **DistilBERT**: 10.7ms vs 13.8ms (1.29x faster)
- **Memory**: 50% reduction across all models
- **Cache**: Instant loading after first run

When you're not benchmarking, you're probably running inference on real data, streaming results, or building something cool. BitInfer handles the optimization so you can focus on the interesting parts.

---

*Built with [heart] for Apple Silicon.*

</details>

---

## Overview

High-performance inference framework optimized for Apple Silicon with Hugging Face integration.

BitInfer delivers up to 1.65x faster inference speeds while reducing memory usage by 50% on M1/M2 processors. The framework provides seamless integration with Hugging Face models through optimized FP16 quantization and intelligent caching.

## Key Features

- **Performance**: 1.65x speedup on BERT-tiny, 1.29x on DistilBERT
- **Memory Efficiency**: 50% reduction in memory footprint
- **Model Caching**: Automatic optimization caching for instant subsequent loads
- **Streaming Interface**: Real-time batch processing with adaptive memory management
- **CLI Support**: Production-ready command-line interface

## Installation

```bash
git clone https://github.com/harpertoken/bitinfer
cd bitinfer
pip install -r requirements.txt
```

## API Reference

### BitInfer Class

#### `BitInfer(model_name: str, device: str = "mps", quantization: str = "pytorch", use_cache: bool = True)`
Initializes BitInfer with optimized model loading and caching.

- **Parameters:**
  - `model_name: str`: Hugging Face model name (e.g., "distilbert-base-uncased")
  - `device: str`: Target device. Default: "mps" (Metal Performance Shaders)
  - `quantization: str`: Optimization method ("pytorch", "metal", "custom"). Default: "pytorch"
  - `use_cache: bool`: Enable model caching for faster subsequent loads. Default: True

- **Returns:** BitInfer: Initialized inference engine with optimized model

#### `infer(text: str) -> torch.Tensor`
Runs single text inference with optimized processing.

- **Parameters:**
  - `text: str`: Input text for inference

- **Returns:** torch.Tensor: Model output with embeddings

#### `batch_infer(texts: List[str]) -> torch.Tensor`
Processes multiple texts in optimized batches.

- **Parameters:**
  - `texts: List[str]`: List of input texts

- **Returns:** torch.Tensor: Batch model outputs

#### `stream_infer(texts: List[str], batch_size: int = None) -> Iterator`
Streams inference results for real-time processing.

- **Parameters:**
  - `texts: List[str]`: Input texts to process
  - `batch_size: int`: Batch size for streaming. Default: 4

- **Returns:** Iterator: Stream of inference results with metadata

#### `adaptive_infer(texts: List[str], max_memory_mb: float = 1000) -> Iterator`
Memory-aware batch processing with automatic size adjustment.

- **Parameters:**
  - `texts: List[str]`: Input texts
  - `max_memory_mb: float`: Maximum memory usage in MB. Default: 1000

- **Returns:** Iterator: Adaptive batch results

#### `cache_info() -> Dict`
Returns model cache information and statistics.

- **Returns:** Dict: Cache size, directory, and metadata

#### `clear_cache() -> None`
Clears all cached optimized models.

### CLI Interface

The `cli.py` command provides comprehensive command-line access to BitInfer functionality.

```
usage: cli.py [-h] [--text TEXT] [--file FILE] [--batch BATCH [BATCH ...]]
              [--quantization {pytorch,metal,custom}] [--device DEVICE]
              [--no-cache] [--output OUTPUT] [--format {json,text}]
              [--streaming] [--batch-size BATCH_SIZE] [--benchmark]
              [--cache-info] [--clear-cache]
              model

BitInfer - Fast inference framework for Apple Silicon

positional arguments:
  model                 Hugging Face model name (e.g., distilbert-base-uncased)

options:
  -h, --help            show this help message and exit
  --text, -t TEXT       Single text to process
  --file, -f FILE       File containing texts (one per line)
  --batch, -b BATCH [BATCH ...]
                        Multiple texts as arguments
  --quantization, -q {pytorch,metal,custom}
                        Quantization method (default: pytorch)
  --device, -d DEVICE   Device to use (default: mps)
  --no-cache            Disable model caching
  --output, -o OUTPUT   Output file for results
  --format {json,text}  Output format
  --streaming, -s       Use streaming inference
  --batch-size BATCH_SIZE
                        Batch size for streaming
  --benchmark           Run benchmark
  --cache-info          Show cache information
  --clear-cache         Clear model cache
```

## Usage

### As a Library

```python
from bitinfer import BitInfer

# Basic usage
model = BitInfer("distilbert-base-uncased")
result = model.infer("Sample text")
print(f"Output shape: {result.last_hidden_state.shape}")
```

#### Advanced Usage

```python
# Batch processing with streaming
model = BitInfer("bert-base-uncased", quantization="pytorch")

texts = ["Text 1", "Text 2", "Text 3", "Text 4"]

# Stream results as they're processed
for result in model.stream_infer(texts, batch_size=2):
    print(f"Processed: {result['text']}")
    print(f"Shape: {result['embedding'].shape}")

# Adaptive memory management
large_texts = ["Long text..."] * 100
for result in model.adaptive_infer(large_texts, max_memory_mb=500):
    process_batch(result)
```

### CLI Usage

```bash
# Single text inference
python cli.py distilbert-base-uncased --text "Hello BitInfer!"

# Batch processing from file
python cli.py bert-base-uncased --file inputs.txt --format json --output results.json

# Streaming with custom batch size
python cli.py prajjwal1/bert-tiny --batch "Text 1" "Text 2" "Text 3" --streaming --batch-size 2

# Performance benchmarking
python cli.py distilbert-base-uncased --text "Benchmark test" --benchmark

# Cache management
python cli.py any-model --cache-info
python cli.py any-model --clear-cache
```

### Advanced Examples

#### Custom Quantization
```python
# Test different optimization methods
models = {
    "pytorch": BitInfer("distilbert-base-uncased", quantization="pytorch"),
    "metal": BitInfer("distilbert-base-uncased", quantization="metal"),
    "custom": BitInfer("distilbert-base-uncased", quantization="custom")
}

for name, model in models.items():
    result = model.infer("Performance test")
    print(f"{name}: {result.last_hidden_state.shape}")
```

#### Error Handling
```python
try:
    model = BitInfer("invalid-model-name")
except Exception as e:
    print(f"Model loading error: {e}")

try:
    result = model.infer("")  # Empty input
except ValueError as e:
    print(f"Input validation error: {e}")
```

## Performance Benchmarks

| Model | BitInfer | PyTorch | Speedup | Memory Reduction |
|-------|----------|---------|---------|------------------|
| BERT-tiny | 4.8ms | 7.9ms | 1.65x | 50% |
| DistilBERT | 10.7ms | 13.8ms | 1.29x | 50% |

## Common Issues

### **1. Model Loading Failures**

Ensure the Hugging Face model name is correct and accessible. BitInfer automatically downloads models on first use.

### **2. Memory Constraints**

On 8GB systems, use smaller models or enable streaming inference for large batches:

```python
# Use streaming for large datasets
for result in model.stream_infer(large_texts, batch_size=2):
    process_result(result)
```

### **3. Device Compatibility**

BitInfer is optimized for Apple Silicon. On other systems, set `device="cpu"`:

```python
model = BitInfer("model-name", device="cpu")
```

### Why Optimization Matters

Proper model optimization is essential for:

* Maximizing inference speed on Apple Silicon
* Reducing memory usage for larger models
* Enabling real-time applications with streaming
* Maintaining accuracy while improving performance

When optimization is bypassed, models may run slower than vanilla PyTorch and consume unnecessary memory.

## Recommendation

For optimal performance:

* Use default `quantization="pytorch"` for best speed/accuracy balance
* Enable caching with `use_cache=True` for repeated model usage
* Use streaming inference for large datasets or real-time applications
* Monitor memory usage with `cache_info()` and clear cache when needed

### Example of Suboptimal Configuration

```python
# Disables key optimizations
model = BitInfer(
    "large-model",
    device="cpu",  # Not using Apple Silicon acceleration
    quantization="custom",  # Slower than pytorch method
    use_cache=False  # No caching benefit
)
```

### Example of Optimal Configuration

```python
# Leverages all optimizations
model = BitInfer(
    "distilbert-base-uncased",
    device="mps",  # Apple Silicon acceleration
    quantization="pytorch",  # Fastest method
    use_cache=True  # Instant subsequent loads
)
```

## Architecture

- **Quantization Engine**: FP16 optimization for Apple Silicon
- **Caching System**: Persistent model optimization storage
- **Streaming Processor**: Memory-aware batch processing
- **CLI Interface**: Full-featured command-line tool

## Requirements

- macOS with Apple Silicon (M1/M2)
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.20+

## License

MIT License
