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
print(model.text2text.infer("The capital of France is", max_new_tokens=14))
# ':\nA) Paris\nB) Lyon\nC) Marseille\nD'
```

## Two APIs, and which one you want

**The high-level API is one door per task, named for the modality pair it maps between.** Every model
carries all of them; the ones it does not answer to say so when called, naming what it actually is.

```python
model.text2text.infer("The capital of France is")        # -> str
model.speech2text.infer(waveform, language="en")         # -> Transcription
model.text2speech.infer("hello world")                   # -> Audio
```

Which door a model answers is read off the file, not guessed from its name:

```python
model.task            # 'automatic-speech-recognition'
model.capabilities    # ('speech2text',)
model.text2speech.infer("hi")
# UnsupportedTask: this model is speech2text (automatic-speech-recognition), not text2speech.
```

Each door does the whole job, including the parts you cannot do from outside. `speech2text` windows
audio for a model whose graph is built at one clip length, decodes with the early stop armed, splits
the output into timestamped segments and **seeks to where the model closed its last segment** — so an
utterance straddling a window edge is re-decoded whole rather than arriving as two fragments:

```python
r = model.speech2text.infer(audio, language="en", timestamps=True)
r.text                        # the joined transcript
r.segments[0].start, .end     # seconds, whole-file
r.timestamped                 # whether those are boundaries the model chose
```

`text2speech` returns audio with its rate attached, because a bare list of floats played at the wrong
rate does not fail — it plays at the wrong speed:

```python
audio = model.text2speech.infer("hello world", steps=8, seed=1)
audio.sample_rate             # 22050
audio.save("out.wav")
```

**The low-level API is the driver's own entry point, and stays raw.** `infer` passes your arguments
straight through, so which ones a model takes is a property of the model rather than of this package:

```python
model.tokenize("The capital of France is")   # [1, 1098, 5706, 803, 4481, 856]
model.detokenize([1, 1098, 5706])            # '<|startoftext|>The capital'
model.infer(tokens=[16, 40, 22, 30], n_steps=4, seed=1234)   # the driver's own inputs
print(model.driver_source)                   # the Lua that will run, and what it accepts
```

Use it when you want a knob the high-level door does not name — VITS's `noise_scale_w`, a specific
voice vector, a model's own second entry point. That is the boundary between the two: **a knob with no
canonical role is reachable through `infer` and nowhere else.**

## Text to speech, and the one step that is not in the file

Every TTS model here takes text now, but two of them get there differently. Supertonic encodes
graphemes itself. The other four consume *phoneme* ids — and the symbol table that turns phonemes into
their ids ships in the GGUF, so `model.tokenizer` is a real vocabulary for all of them:

```python
model.tokenizer                      # <loom.Tokenizer 'phonemes' size=159>
model.tokenize("h\u0259\u02c8lo\u028a")             # -> ids, with the model's own BOS/blank/EOS assembly
```

What is *not* in the file is grapheme-to-phoneme, because that is a property of the language rather
than of any checkpoint. It is an optional extra:

```sh
pip install "loom-py-rt[phonemes]"
```

With it, `text2speech.infer("hello world")` works on all five. Without it, passing `phonemes=` or
`tokens=` works exactly as before and only the text door is absent, with an error naming the install.
`loom.phonemizers.register("ipa", fn)` substitutes your own.

## Choosing a device

A wheel built with a GPU backend uses it by default; one built without has only a CPU to find, so
nothing changes.

```python
model = loom.Model.from_file("qwen3.gguf")                  # decide for me (or $LOOM_DEVICE)
model = loom.Model.from_file("qwen3.gguf", device="cpu")    # pin it
model = loom.Model.from_file("qwen3.gguf", device="gpu")    # demand one; raises if there is none
model.device, model.device_description   # ('Vulkan0', 'AMD Radeon Vega 3 Graphics (RADV RAVEN2)')
```

`"gpu"` raises rather than falling back, because a caller who spelled it out is asking a question
about the machine and a silent CPU run is how a large slowdown goes unnoticed. `"auto"` — the default
— is the one that falls back.

The base wheel is CPU-only, and an accelerator is a separate install rather than a different wheel:

```sh
pip install "loom-py-rt[vulkan]"
```

That adds one small package holding one `libggml-vulkan.so`, which this package finds at import;
`device="auto"` then uses it and nothing about the base wheel changes. The reason it works this way —
rather than a full wheel per accelerator, which is the more familiar shape — is that a Vulkan backend
is 46.5 MB and CUDA is larger, so the per-accelerator matrix does not fit PyPI's 100 MB per-file
ceiling. See [`packaging/README.md`](packaging/README.md).

```python
loom.devices()   # [{'name': 'Vulkan0', 'description': 'AMD Radeon Vega 3 Graphics (RADV RAVEN2)', ...}]
```

Worth calling after installing one, because a backend whose driver is too old — or which finds no
supported device — loads without error and registers nothing, and the only other symptom is a model
running at CPU speed. Note that with this build **every** backend is loaded at run time, the CPU
included, so an empty device list means no backend library was found at all rather than no accelerator.

Which ops fall back to the CPU, and why some always will, is documented in
[loom.cpp's own build notes](https://github.com/loom-ai-org/loom.cpp#running-on-a-gpu).

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

These are the models `text2text` answers to; `model.generate(...)` is the same call under the name
it shipped with.

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

`model.speech2text.infer(waveform, language="en")` — the mel frontend is inside the graph, so a raw
waveform is the input, and long audio is windowed and seeked for you.

### Speech synthesis

| Model | Exported from |
|---|---|
| [`loom-ai-org/kokoro-82m-loom`](https://huggingface.co/loom-ai-org/kokoro-82m-loom) | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) |
| [`loom-ai-org/matcha-tts-ljspeech-loom`](https://huggingface.co/loom-ai-org/matcha-tts-ljspeech-loom) | [Matcha-TTS (LJSpeech checkpoint)](https://github.com/shivammehta25/Matcha-TTS) |
| [`loom-ai-org/supertonic-2-loom`](https://huggingface.co/loom-ai-org/supertonic-2-loom) | [`Supertone/supertonic-2`](https://huggingface.co/Supertone/supertonic-2) |
| [`loom-ai-org/vits-piper-en-gb-miro-loom`](https://huggingface.co/loom-ai-org/vits-piper-en-gb-miro-loom) | [`OpenVoiceOS/pipertts_en-GB_miro`](https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro) |
| [`loom-ai-org/styletts2-ljspeech-loom`](https://huggingface.co/loom-ai-org/styletts2-ljspeech-loom) | [`yl4579/StyleTTS2-LJSpeech`](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) |

All five take text through `model.text2speech.infer(...)`; the four phoneme-input ones need the
`[phonemes]` extra for the G2P step (see above), and every one of them accepts `phonemes=` or `tokens=`
without it. Kokoro and Supertonic ship a default voice, so a published file speaks on its own.
`model.driver_source` is the authority on what each driver accepts.

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

**1. GPUs and NPUs — the packaging is built; the backends beyond Vulkan are what remain.** The engine
schedules a graph across a device backend and a CPU fallback, this package exposes the choice as
`device=` (above), and the wheel shape that lets an accelerator ship at all now exists.

The shape that is NOT wanted is a wheel per accelerator per architecture — PyPI's wheel tags have no
accelerator dimension, so that is torch's `cu121` arrangement, and it multiplies every future backend by
every existing platform. `GGML_BACKEND_DL` makes the better shape possible and this package is now
built that way: **one arch-tagged base wheel, plus small backend packages that drop a `.so` where ggml
looks for it**, so `pip install "loom-py-rt[cuda]"` means "also fetch that backend", `device="auto"`
finds it, and a Raspberry Pi installs nothing extra. `packaging/rt-vulkan/` is the worked example; a
CUDA package is that directory with two strings changed, waiting only on a machine with an NVIDIA GPU
to build and test against. Tracked as `BACKLOG.md` P4.8, along with which backends are
reachable at all (CUDA, OpenVINO and Qualcomm's are already in the pinned ggml; CoreML and RKNPU2 are
not).

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
