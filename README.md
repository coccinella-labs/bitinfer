<p align="center">
  <img src="https://raw.githubusercontent.com/coccinella-labs/bitinfer/main/.github/assets/thumbnail.png" alt="bitinfer" width="100%">
</p>

# BitInfer

Hugging Face encoder inference for Apple Silicon, with float16 weights, on-disk weight caching, streaming, and a CLI.

Read this before trusting a performance claim. The previous version of this README advertised 1.65x speedups and 50% memory reduction. The memory number was real. **The speedup was not.** On CPU this package is about 35% slower than plain `transformers`, because float16 weights are upcast on every matmul and nothing compensates for it. There are no MPS measurements. Every number below comes from `bench.py`, which is in the repository and reproduces them.

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
| model                       | BitInfer | transformers | ratio | memory MB   |
|-----------------------------|----------|--------------|-------|-------------|
| tiny-random-bert            | 1.65ms   | 1.06ms       | 0.64x | 0.3 -> 0.2  |
| bert_uncased_L-2_H-128_A-2  | 0.84ms   | 0.54ms       | 0.64x | 16.7 -> 8.4 |
```

Ratio is transformers divided by BitInfer, so below 1.00x means BitInfer is **slower**.

- **Memory halves.** 16.7 MB to 8.4 MB is float32 to float16, exactly as expected. This is the one solid win, and it is a dtype change rather than anything clever.
- **BitInfer is slower, by about 35%.** Float16 weights are not free on CPU: they are upcast per matmul and there is no fused kernel to make up for it. Ad-hoc measurements on this machine ranged from 0.78x to 1.15x depending on load, which is why `bench.py` repeats the comparison and takes medians. Even the favourable readings were noise.

So this package reduces memory and costs speed on CPU. Reproduce with `python bench.py`.

## Limitations

- The comparison is CPU. No MPS numbers are published here because none were measured.
- Models must be loadable by `AutoModel`. Causal LMs and encoder-decoder models are out of scope.
- `adaptive_infer` estimates memory from text length. It does not query available RAM.
- No batch-size tuning against measured throughput.
- `cpp/`, `metal/` and `bindings/` are not wired into the Python path. The Python implementation is pure PyTorch.

## Layout

```
cli.py     command line entry point
bench.py   reproduces the measured table above
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
