<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-inline-dark.svg">
    <img src="assets/logo-inline.svg" alt="" width="52" align="middle">
  </picture>
  &nbsp;loom-py
</h1>

```python
import loom

model = loom.Model.from_pretrained("loom-ai-org/lfm2-350m-monolithic-loom")
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

## Supported models

Seventeen, published at [huggingface.co/loom-ai-org](https://huggingface.co/loom-ai-org) and loadable
by id with `from_pretrained` (needs the `[hub]` extra). This package has no per-architecture code, so
the list is a property of [loom-exporter](https://github.com/loom-ai-org/loom-exporter), not of
anything here.

### Language models

| Model | Exported from |
|---|---|
| [`loom-ai-org/qwen3-0.6b-base-loom`](https://huggingface.co/loom-ai-org/qwen3-0.6b-base-loom) | [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) |
| [`loom-ai-org/lfm2-350m-monolithic-loom`](https://huggingface.co/loom-ai-org/lfm2-350m-monolithic-loom) | [`LiquidAI/LFM2-350M`](https://huggingface.co/LiquidAI/LFM2-350M) |
| [`loom-ai-org/lfm2-350m-modular-loom`](https://huggingface.co/loom-ai-org/lfm2-350m-modular-loom) | [`LiquidAI/LFM2-350M`](https://huggingface.co/LiquidAI/LFM2-350M) |
| [`loom-ai-org/smollm2-360m-instruct-loom`](https://huggingface.co/loom-ai-org/smollm2-360m-instruct-loom) | [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| [`loom-ai-org/gemma-3-270m-it-loom`](https://huggingface.co/loom-ai-org/gemma-3-270m-it-loom) | [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) |

These are the models `generate` works on.

### Speech recognition

| Model | Exported from |
|---|---|
| [`loom-ai-org/whisper-small-loom`](https://huggingface.co/loom-ai-org/whisper-small-loom) | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) |
| [`loom-ai-org/conformer-ctc-small-loom`](https://huggingface.co/loom-ai-org/conformer-ctc-small-loom) | [`nvidia/stt_en_conformer_ctc_small`](https://huggingface.co/nvidia/stt_en_conformer_ctc_small) |
| [`loom-ai-org/parakeet-tdt-0.6b-loom`](https://huggingface.co/loom-ai-org/parakeet-tdt-0.6b-loom) | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| [`loom-ai-org/parakeet-rnnt-0.6b-loom`](https://huggingface.co/loom-ai-org/parakeet-rnnt-0.6b-loom) | [`nvidia/parakeet-rnnt-0.6b`](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) |
| [`loom-ai-org/gigaam-v3-rnnt-loom`](https://huggingface.co/loom-ai-org/gigaam-v3-rnnt-loom) | [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3) |
| [`loom-ai-org/qwen3-asr-0.6b-loom`](https://huggingface.co/loom-ai-org/qwen3-asr-0.6b-loom) | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| [`loom-ai-org/granite-speech-4.0-1b-loom`](https://huggingface.co/loom-ai-org/granite-speech-4.0-1b-loom) | [`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) |

`model.detokenize(model.infer(waveform=audio, audio_samples=len(audio)))` — the mel frontend is inside
the graph, so a raw waveform is the input.

### Speech synthesis

| Model | Exported from |
|---|---|
| [`loom-ai-org/kokoro-82m-loom`](https://huggingface.co/loom-ai-org/kokoro-82m-loom) | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) |
| [`loom-ai-org/matcha-tts-ljspeech-loom`](https://huggingface.co/loom-ai-org/matcha-tts-ljspeech-loom) | [Matcha-TTS (LJSpeech checkpoint)](https://github.com/shivammehta25/Matcha-TTS) |
| [`loom-ai-org/supertonic-2-loom`](https://huggingface.co/loom-ai-org/supertonic-2-loom) | [`Supertone/supertonic-2`](https://huggingface.co/Supertone/supertonic-2) |
| [`loom-ai-org/vits-piper-en-gb-miro-loom`](https://huggingface.co/loom-ai-org/vits-piper-en-gb-miro-loom) | [`OpenVoiceOS/pipertts_en-GB_miro`](https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro) |
| [`loom-ai-org/styletts2-ljspeech-loom`](https://huggingface.co/loom-ai-org/styletts2-ljspeech-loom) | [`yl4579/StyleTTS2-LJSpeech`](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) |

Supertonic is the one with a text door — `model.tokenize` works on it and is `None` for the other
four, which take phoneme ids (see above). `model.driver_source` is the authority on what each accepts.

## The three repos

| | |
|---|---|
| [**loom.cpp**](https://github.com/loom-ai-org/loom.cpp) | the engine, vendored here as a submodule |
| [**loom-exporter**](https://github.com/loom-ai-org/loom-exporter) | produces the GGUFs this runs |
| [**loom-py**](https://github.com/loom-ai-org/loom-py) | this one |

## Installing

```sh
pip install loom-py-rt           # once published -- `loom-py` on PyPI clashes with `loompy`,
                                  # and `loom-engine` normalizes to the already-taken `loomengine`
pip install loom-py-rt[hub]      # + from_pretrained()
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

## Roadmap

Shared with [loom.cpp](https://github.com/loom-ai-org/loom.cpp), because three of the four are the
engine's and this package inherits them by having no per-architecture code of its own.

**1. GPUs and NPUs.** The engine talks to a single `ggml_backend_t` and uses no `ggml_backend_sched`,
which is what has to change before a second device can hold part of a graph. Nothing in this package
should need to change with it — device selection is a property of how the engine is built.

**2. Wheels for more platforms.** Linux x86-64 today; next macOS on Intel, macOS on Apple Silicon and
Linux on ARM. This is the item most visible from here, since it is what `pip install loom-py-rt` can
resolve to.

**3. More models — P5 in the ledger**, ordered by coverage per unit of effort: BERT token classifiers
(the smallest possible template, and the first non-audio task) → codec decoders → CNN+CTC and SANM
encoders → the remaining TTS families → text encoder-decoders → small classifiers → music. Each lands
here for free: a model this package has never heard of works the day the exporter can produce it.

**4. The follow-ups the docs already name** —
[`BACKLOG.md`](https://github.com/loom-ai-org/loom.cpp/blob/main/BACKLOG.md) is the ledger for all three
repos and the authority. The ones that would show up in this API: a permissively-licensed phonemiser,
which is what would give the four phoneme-input TTS models a `tokenize`; and the `KvCache` memory
redesign and quantized KV cache, which decide how large a model this can run on a given machine.

## Licence

MIT — see [`LICENSE`](LICENSE).
