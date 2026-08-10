<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="loom.cpp" width="96">
  </picture>
</p>

<h1 align="center">loom-py</h1>

<p align="center"><em>Python bindings for loom.cpp</em></p>

```python
import loom

model = loom.Model.from_pretrained("femelo/matcha-tts-loom")
audio = model.infer(tokens=[16, 40, 22, 30, 12, 3, 25, 19, 44, 11, 2], n_steps=4, seed=1234)
```

## Why there is so little API

A loom GGUF carries its own graph topologies and its own driver script alongside its weights, so this
package contains **no per-architecture code at all**. Loading a model registers whatever topologies
the file declares and attaches a KV cache to the ones that say they need it; running one calls the
driver the file shipped with. A model this library has never heard of works the day
[loom-exporter](https://github.com/femelo/loom-exporter) can produce it.

That is also why `infer` takes `**kwargs`: its arguments are the *driver's* arguments, and which ones
a model takes is a property of the model. `model.driver_source` prints the Lua that will run, whose
header comment documents its inputs — that is the authority.

```python
model = loom.Model.from_file("granite_speech_mil.gguf")
model.architecture          # 'granite-speech'
model.topologies            # ['encoder', 'embed', 'decoder', 'lm_head']
model.hparam("samples_per_chunk")   # 192000
print(model.driver_source)  # what infer() will run, and what it accepts
```

## The three repos

| | |
|---|---|
| [**loom.cpp**](https://github.com/femelo/loom.cpp) | the engine, vendored here as a submodule |
| [**loom-exporter**](https://github.com/femelo/loom-exporter) | produces the GGUFs this runs |
| [**loom-py**](https://github.com/femelo/loom-py) | this one |

## Installing

```sh
pip install loom-py           # once published
pip install loom-py[hub]      # + from_pretrained()
```

From a checkout — note `--recursive`, since the engine is a submodule:

```sh
git clone --recursive https://github.com/femelo/loom-py
cd loom-py && pip install -e .
```

No runtime dependencies. Arrays cross the boundary as plain sequences of floats, so numpy is something
you may use rather than something this package makes you install — `list`, `array.array`, numpy arrays
and torch tensors all work.

## Testing

```sh
pytest tests/ci      # the Python layer: coercion and error paths. No model. What CI runs.
pytest tests/gate    # a real exported GGUF, end to end.
```

```sh
export LOOM_TEST_MODEL=~/loom-fixtures/matcha_mil.gguf
export LOOM_TEST_MODEL_INPUTS='{"tokens":[16,40,22,30,12,3],"n_steps":4,"seed":1234}'
pytest tests/gate -q
```

The gate suite is written against no particular architecture on purpose: it asserts the *shape* of
what a loom model is, which is the whole of what this package knows. A test that expected one model's
inputs would be this package learning about a model, which is the thing the design exists to avoid.

## Licence

MIT — see [`LICENSE`](LICENSE).
