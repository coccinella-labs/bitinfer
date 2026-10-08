<p align="center">
  <img src="https://raw.githubusercontent.com/coccinella-labs/bitinfer/main/.github/assets/thumbnail.png" alt="bitinfer" width="100%">
</p>

# BitInfer

Hugging Face encoder inference for Apple Silicon, with float16 weights, on-disk weight caching, streaming, and a CLI.

Read this before trusting a performance claim, because the honest answer depends entirely on model size. On a model large enough for the arithmetic to dominate, float16 weights are a large win: about **2.3x faster and half the memory** on distilbert-base-uncased. On a model too small to be dominated by arithmetic, it is a **loss**: about 35% slower, because the per-call overhead of upcasting outweighs work that was already free.

The previous version of this README quoted a single 1.65x figure with no model size or hardware attached. That number was not reproducible. Every figure below comes from `bench.py`, which is in the repository and regenerates the table. There are no MPS measurements.

## Capability

Verified by running each row against this checkout, not read off the source.

| Capability                     | BitInfer |
| ------------------------------ | -------- |
| Load Hugging Face model        | **Yes**  |
| Tokenization                   | **Yes**  |
| CPU/MPS execution              | **Yes**  |
| Batch inference                | **Yes**  |
| Streaming inference            | **Yes**  |
| Adaptive batching              | **Yes**  |
| Model caching                  | **Yes**  |
| Quantization/optimization path | **Yes**  |
| Training                       | **No**   |
| Backpropagation                | **No**   |
| Optimizer                      | **No**   |
| Fine-tuning                    | **No**   |
| Distributed training           | **No**   |

Notes on the rows that are easy to misread:

- **MPS is the default device** and the tensors do land on the GPU, but every
  performance number below is CPU. MPS runs; MPS speed is unmeasured.
- **Quantization path Yes** does not mean 8-bit. `quantization="pytorch"` calls
  `quantize_dynamic(dtype=torch.qint8)`, which raises on Apple Silicon, and the
  handler falls back to `model.half()`. The path exists and works; it ends in
  float16.
- The five **No** rows are structural. There is no `torch.optim`, no `.backward()`,
  no `requires_grad`, and no `torch.distributed` anywhere in the tree, so there is
  no training code to switch off.

## Install

```bash
git clone https://github.com/coccinella-labs/bitinfer
cd bitinfer
pip install -r requirements.txt
```

Apple Silicon is the target. `device="mps"` is the default; `device="cpu"` works everywhere and is what the numbers below were taken on, because a CPU comparison is the honest one to make.

## Usage

The package directory is `python/`, so imports need a path entry:

```python
import sys
sys.path.insert(0, "python")

from core import BitInfer

model = BitInfer("google/bert_uncased_L-2_H-128_A-2")
result = model.infer("Sample text")

print(result.last_hidden_state.shape)   # (1, sequence_length, hidden)
```

`infer` returns the Hugging Face model output, not a tensor, so the embeddings live at `.last_hidden_state`. After `pip install -e .` the modules import directly and the path line is unnecessary.

### Streaming

```python
for item in model.stream_infer(["first", "second"], batch_size=2):
    item["text"]        # the input string
    item["embedding"]   # last_hidden_state[row] for that input
    item["batch_index"] # position in the original list
```

Batches internally and yields one dict per input. Verified keys: `batch_index`, `embedding`, `text`.

### Adaptive batching

```python
for item in model.adaptive_infer(texts, max_memory_mb=500):
    ...
```

Batch size is estimated from average token count over the first 10 inputs, capped at 32. It is a heuristic on text length, not a measurement of available memory.

### Caching

```python
model = BitInfer(name, use_cache=True)
model.cache_info()   # {'cache_size_mb': ..., 'cache_dir': '~/.bitinfer_cache'}
model.clear_cache()
```

The cache stores a `state_dict` under `~/.bitinfer_cache`, keyed by model name and quantization method.

Caching had a real bug, now fixed. `load_state_dict` copies each tensor into the destination parameter's dtype, and the template comes from `from_pretrained` in float32. A model cached as float16 was therefore silently upcast to float32 on the next run, so the "optimized" model ran unoptimized while reporting a cache hit. `load_model` now restores the cached dtype:

```
before   1st run float16   cache hit float32   <- optimization lost
after    1st run float16   cache hit float16
```

### CLI

```bash
python cli.py <model> --text "Hello"
python cli.py <model> --file inputs.txt --format json --output results.json
python cli.py <model> --batch "one" "two" --streaming --batch-size 2
python cli.py <model> --cache-info
python cli.py <model> --clear-cache
python cli.py <model> --text "..." --benchmark
```

`--benchmark` times a single run of the current code against nothing. There is no baseline comparison anywhere in it, so its output cannot support a speedup claim. Use `bench.py` for that.

## What quantization actually does

`quantization="pytorch"` is the default and its name is misleading. The path is:

```python
torch.quantization.quantize_dynamic(model, {nn.Linear, ...}, dtype=torch.qint8)
```

On Apple Silicon that call raises `RuntimeError` containing `NoQEngine`, and the handler falls back to `model.half()`. So on this hardware the default path ends up float16, not 8-bit, and it reaches that state through an exception rather than by design.

`quantization="metal"` casts inputs to half inside `infer`. `quantization="custom"` uses an 8-bit implementation in `python/custom_quant.py`.

## Measured results

Apple M1, arm64, Python 3.14, torch 2.14.1, `device="cpu"`, median of 40 timed calls after warmup, whole comparison repeated three times.

```
| model                       | BitInfer | transformers | ratio | memory MB        |
|-----------------------------|----------|--------------|-------|------------------|
| tiny-random-bert            | 1.66ms   | 1.08ms       | 0.65x | 0.3 -> 0.2       |
| bert_uncased_L-2_H-128_A-2  | 0.84ms   | 0.54ms       | 0.64x | 16.7 -> 8.4      |
| distilbert-base-uncased     | 6.77ms   | 15.61ms      | 2.30x | 253.2 -> 126.6   |
```

Ratio is transformers divided by BitInfer, so above 1.00x means BitInfer is faster.

- **Memory halves on every model.** That is float32 to float16 and it is unconditional. It is a dtype change, not anything clever.
- **Speed splits on model size.** The two small models are around 0.3 MB and 17 MB, where a call is mostly Python and dispatch overhead; the float16 upcast is pure added cost and BitInfer loses by about a third. distilbert-base-uncased is 66M parameters, the arithmetic dominates, and float16 wins by about 2.3x.
- **The crossover is somewhere between 17 MB and 253 MB of weights.** It was not located precisely, because doing so properly needs a sweep rather than three points.

The small-model rows are also why an earlier version of this file reported a flat "0.64x, slower". That was a real measurement of two toy models, presented as a general result. Reproduce all three with `python bench.py`.

## Limitations

- The comparison is CPU. No MPS numbers are published here because none were measured.
- Speed depends on model size, and the crossover is not precisely located. Do not assume a speedup on a model smaller than those measured here.
- Models must be loadable by `AutoModel`. Causal LMs and encoder-decoder models are out of scope.
- `adaptive_infer` estimates memory from text length. It does not query available RAM.
- No batch-size tuning against measured throughput.
- `cpp/`, `metal/` and `bindings/` are not wired into the Python path. The Python implementation is pure PyTorch.

## Layout

```
cli.py     command line entry point
bench.py   reproduces the table above across three model sizes
tests/     pytest suite, including the cache dtype regression
examples/  usage scripts

python/    the implementation
  core.py           BitInfer
  quantization.py   dynamic quantize, float16 fallback
  cache.py          state_dict cache
  streaming.py      streaming and adaptive batching
  custom_quant.py   8-bit path, quantization="custom"
  metal_opt.py      FP16 input cast, quantization="metal"

Not used by the Python path, which is pure PyTorch:
cpp/       C++ sources, unused
metal/     Metal shader, unused
bindings/  pybind11 binding, unused
```

## Tests

```bash
python -m pytest tests/ -v
```

CI runs the suite on macOS and lint on Ubuntu. Reverting the cache fix makes `test_cache_preserves_dtype` fail, so that bug cannot return silently.

## Requirements

Python 3.9+, torch 2.0+, transformers 4.20+, Apple Silicon for the default device.

## License

MIT. See [LICENSE](LICENSE).
