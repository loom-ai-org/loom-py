<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-inline-dark.svg">
    <img src="assets/logo-inline.svg" alt="" width="52" align="middle">
  </picture>
  &nbsp;loom-py
</h1>

```python
import loom

model = loom.Model.from_pretrained("loom-ai-org/lfm2-350m-loom")
print(model.generate("The capital of France is", max_new_tokens=14))
# ':\nA) Paris\nB) Lyon\nC) Marseille\nD'
```

## Text in, text out

`generate` tokenizes with the vocabulary the GGUF embeds, runs the driver, and detokenizes what comes
back. The same steps are available separately when you want them:

```python
model.tokenize("The capital of France is")   # [1, 1098, 5706, 803, 4481, 856]
model.detokenize([1, 1098, 5706])            # '<|startoftext|>The capital'
model.tokenizer                               # <loom.Tokenizer 'gpt2' size=64400>
```

The four vocabulary families a loom GGUF can carry — byte-level BPE, SentencePiece, WordPiece and
byte-level — are dispatched on the file's own `tokenizer.ggml.model`, so this is one call whichever
one a model uses.

For a speech model there is nothing to encode; detokenizing the driver's output is the other half of
the same thing:

```python
transcript = model.detokenize(model.infer(waveform=audio, audio_samples=len(audio)))
```

**A TTS model has no `generate`, and that is a real limitation rather than a missing feature.** Matcha,
VITS, Kokoro and StyleTTS2 consume *phoneme* ids that a phonemiser produces outside the engine, so
their GGUFs embed no vocabulary at all — `model.tokenizer` is `None` for them and they take ids
directly:

```python
audio = model.infer(tokens=[16, 40, 22, 30, 12, 3], n_steps=4, seed=1234)
```

## Why there is so little API

A loom GGUF carries its own graph topologies and its own driver script alongside its weights, so this
package contains **no per-architecture code at all**. Loading a model registers whatever topologies
the file declares and attaches a KV cache to the ones that say they need it; running one calls the
driver the file shipped with. A model this library has never heard of works the day
[loom-exporter](https://github.com/loom-ai-org/loom-exporter) can produce it.

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
| [**loom.cpp**](https://github.com/loom-ai-org/loom.cpp) | the engine, vendored here as a submodule |
| [**loom-exporter**](https://github.com/loom-ai-org/loom-exporter) | produces the GGUFs this runs |
| [**loom-py**](https://github.com/loom-ai-org/loom-py) | this one |

## Installing

```sh
pip install loom-py           # once published
pip install loom-py[hub]      # + from_pretrained()
```

From a checkout — note `--recursive`, since the engine is a submodule:

```sh
git clone --recursive https://github.com/loom-ai-org/loom-py
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
