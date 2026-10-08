# Contributing to BitInfer

## Setup

```bash
git clone https://github.com/coccinella-labs/bitinfer
cd bitinfer
pip install -r requirements.txt
```

## Before you push

```bash
pre-commit install
pre-commit run --all-files
python -m pytest tests/ -v
```

Pre-commit is not advisory. Its black, isort and flake8 hooks rewrite files in
place, and CI fails when they modify anything. Running it locally is how you find
out before CI does.

CI runs four Python versions for lint, the suite and a smoke test on macOS, and
builds the package on main pushes only.

## Layout

The package directory is `python/`, not a directory named `bitinfer`. Scripts
work around this with `sys.path.insert`; `pip install -e .` makes the modules
importable directly.

`cpp/`, `metal/` and `bindings/` are not wired into the Python path. The Python
implementation is pure PyTorch. Do not assume a change there affects anything.

## Things worth knowing before you change them

**The cache stores a `state_dict`, not a model.** `load_state_dict` copies each
tensor into the destination parameter's dtype, and the template comes from
`from_pretrained` in float32. A model cached as float16 was silently upcast to
float32 on the next run, so the optimized model ran unoptimized while reporting
a cache hit. `load_model` therefore restores the cached dtype after loading.
`tests/test_cache_dtype.py` fails if that regresses.

**`quantization="pytorch"` does not give you 8-bit on Apple Silicon.** It calls
`quantize_dynamic(dtype=torch.qint8)`, which raises, and the handler falls back
to `model.half()`. If you change that path, update the README, because the
capability table and the quantization section both describe it.

**Performance claims need evidence.** Run `python bench.py` and quote what it
prints, including the model size. Results split on model size: roughly 2.3x on
distilbert-base-uncased and a third slower on models small enough to be dominated
by dispatch overhead. All numbers are CPU; MPS runs but is unmeasured.

The README's capability table is verified by running each row, not by reading the
source. If you change what the package can do, update that table too.

## Adding a benchmark or a test

Tests go in `tests/` and are plain pytest. `bench.py` prints the table the README
quotes; add a model to its `MODELS` list and give it a call count in `CALLS_FOR`
if it needs more iterations to produce a stable median.

## License

MIT. See [LICENSE](LICENSE).
