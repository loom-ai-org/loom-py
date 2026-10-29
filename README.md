<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/loom-ai-org/loom-py/main/assets/logo-inline-dark.svg">
    <img src="https://raw.githubusercontent.com/loom-ai-org/loom-py/main/assets/logo-inline.svg" alt="" width="52" align="middle">
  </picture>
  &nbsp;loom-py
</h1>

<p align="center">
  <b>Speech, audio and language models in a few lines of Python, from a GPU down to a Raspberry Pi Zero.</b><br>
  Python bindings for <a href="https://github.com/loom-ai-org/loom.cpp">loom.cpp</a> · published on PyPI as <code>loom-py-rt</code>
</p>

<p align="center">
  <a href="https://pypi.org/project/loom-py-rt/"><img alt="PyPI" src="https://img.shields.io/pypi/v/loom-py-rt?include_prereleases&color=0969da"></a>
  <img alt="Python 3.10 to 3.13" src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-3776ab">
  <a href="https://huggingface.co/loom-ai-org"><img alt="Models on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20models-64-ffcc4d"></a>
  <a href="https://github.com/loom-ai-org/loom-py/blob/main/LICENSE"><img alt="MIT licence" src="https://img.shields.io/badge/licence-MIT-1a7f37"></a>
</p>

---

```python
import loom

tts = loom.Model.from_pretrained("loom-ai-org/pocket-tts-loom", "pocket-tts-q8_0_pc.gguf")
tts.text2speech.infer("Hello from a Raspberry Pi!").save("hello.wav")
```

* **64 ready-to-run models** on [Hugging Face](https://huggingface.co/loom-ai-org): speech recognition,
  text to speech, voice activity, speaker and language id, music, codecs and small language models.
* **One call per task**, the same for every model that does it, with long audio, timestamps, voices
  and chat templates handled for you.
* **Fast on small boards.** On a Raspberry Pi 4, loom's int8 is 1.2x to 9.3x faster than onnxruntime's.
  On a Pi Zero, wake-word features, VAD, tiny TTS voices and Citrinet ASR run faster than real time,
  and its GPU's 12 QPUs make them 1.1x to 3.4x faster still.
* **No runtime dependencies.** Audio goes in as any sequence of floats: a `list`, a NumPy array or a
  torch tensor.

## Install

```sh
pip install "loom-py-rt[hub]"          # + from_pretrained() from the Hugging Face Hub
```

| Extra | Adds |
|---|---|
| `[hub]` | `loom.Model.from_pretrained(...)` |
| `[phonemes]` | text input for the models that take phonemes (Kokoro, Piper VITS, sanoTTS, Vocetta2, Matcha, StyleTTS2) |
| `[vulkan]` · `[cuda]` · `[metal]` | a GPU backend, as a small add-on package |

Wheels cover Linux x86-64 and ARM64 (Raspberry Pi 3/4/5 on 64-bit Raspberry Pi OS, Jetson, Graviton),
and macOS on Apple Silicon and Intel, for CPython 3.10 to 3.13. The **Raspberry Pi Zero / Zero W / Pi 1**
wheel is a release asset, because PyPI does not accept ARMv6 wheels:

<!-- The version in the wheel filename below is derived from the repo-root `VERSION` file by
     `python packaging/version.py --set <version>`; tests/ci/test_version_consistency.py fails
     if it drifts. `releases/latest/download/` is stable, the filename is not. -->
```sh
pip install https://github.com/loom-ai-org/loom-py/releases/latest/download/loom_py_rt-1.0.0rc19-cp311-cp311-linux_armv6l.whl
```

Windows is not supported yet. Which CPU kernels a machine gets, and every platform detail, is in
[docs/PLATFORMS.md](https://github.com/loom-ai-org/loom-py/blob/main/docs/PLATFORMS.md).

## What you can do with it

Every model answers through a **door** named for what goes in and what comes out. A model carries all
of them, and the ones it doesn't do say so, naming what it does.

### 🗣️ Text to speech

```python
tts = loom.Model.from_pretrained("loom-ai-org/pocket-tts-loom", "pocket-tts-q8_0_pc.gguf")
audio = tts.text2speech.infer("The quick brown fox jumps over the lazy dog.")
audio.sample_rate, audio.duration      # (24000, 3.04)
audio.save("fox.wav")

tts.voices                             # ['alba', 'anna', 'azelma', 'bill_boerst', ...]
tts.text2speech.infer("Good morning.", voice="marius").save("marius.wav")
```

### 🎙️ Speech recognition

```python
import soundfile
audio, _ = soundfile.read("speech.wav")            # 16 kHz mono

asr = loom.Model.from_pretrained("loom-ai-org/whisper-small-loom")
result = asr.speech2text.infer(audio, language="en", timestamps=True)
result.text
for segment in result.segments:
    print(segment.start, segment.end, segment.text)
```

Long recordings are windowed for you, and each window starts where the model closed its last segment,
so a sentence on a window edge comes back whole.

### 💬 Language models

```python
lm = loom.Model.from_pretrained("loom-ai-org/gemma-3-270m-it-loom")
lm.text2text.chat("What is the capital of France?", temperature=0.0)
# 'The capital of France is Paris.'

base = loom.Model.from_pretrained("loom-ai-org/qwen3-0.6b-base-loom")
base.text2text.infer("The capital of France is", max_new_tokens=16)   # a plain completion
```

Decoding follows each checkpoint's own generation settings. Pass `temperature`, `top_k`, `top_p` or
`seed` to override them.

### 🎧 Voice activity, language and speaker id

```python
vad = loom.Model.from_pretrained("loom-ai-org/silero-vad-loom")
frames = vad.speech2class.infer(audio)            # one row per 32 ms frame
speech = frames.probability("speech")             # the speech curve, yours to threshold
frames.times[:3]                                  # [0.0, 0.032, 0.064]

lid = loom.Model.from_pretrained("loom-ai-org/ecapa-voxlingua107-loom")
lid.speech2class.infer(audio).top(3)              # [('en: English', 0.82), ('la: Latin', 0.03), ...]

spk = loom.Model.from_pretrained("loom-ai-org/titanet-large-loom")
embedding = spk.speech2embeddings.infer(audio)    # 192 floats; compare two with cosine similarity
```

### 🏷️ Text classification

```python
ner = loom.Model.from_pretrained("loom-ai-org/distilbert-ner-loom")
[(t.piece, t.label) for t in ner.text2class.infer("Wolfgang lives in Berlin")]
# [('Wolfgang', 'B-PER'), ('lives', 'O'), ('in', 'O'), ('Berlin', 'B-LOC')]
```

### 🎵 Codec models: dialogue, music and voice cloning

Some models write codec tokens, and a second file turns those into audio:

```python
dia = loom.Model.from_pretrained("loom-ai-org/dia-1.6b-loom")
dac = loom.Model.from_pretrained("loom-ai-org/dac-44khz-loom")
codes = dia.text2codes.infer("[S1] Hello world.")
dac.codes2speech.infer(codes).save("dialogue.wav")
```

MusicGen + EnCodec, Qwen3-TTS + its tokenizer and MOSS-TTS + its codec pair the same way.

### 🔧 The layer underneath

Each file embeds its own driver script, and `model.infer(...)` passes your arguments straight to it,
for the knobs a door doesn't name:

```python
vits = loom.Model.from_pretrained("loom-ai-org/vits-piper-en-gb-miro-loom")
vits.task, vits.capabilities       # ('text-to-speech', ('text2speech',))
print(vits.driver_source)          # the driver, with a header documenting every argument it takes
samples = vits.infer(tokens=vits.tokenize("həˈloʊ"), seed=1, noise_scale_w=0.8)
```

[docs/GUIDE.md](https://github.com/loom-ai-org/loom-py/blob/main/docs/GUIDE.md) covers every door in
depth: what each returns, and why.

## Models

Every model on [huggingface.co/loom-ai-org](https://huggingface.co/loom-ai-org) loads by id. A few to
start with:

| Task | Door | Try | Also |
|---|---|---|---|
| Speech recognition | `speech2text` | [whisper-small](https://huggingface.co/loom-ai-org/whisper-small-loom) (99 languages) | Parakeet, Canary, Nemotron, Qwen3-ASR, SenseVoice, Moonshine, Citrinet, ... (20) |
| Text to speech | `text2speech` | [Pocket TTS](https://huggingface.co/loom-ai-org/pocket-tts-loom) (26 voices) | Kokoro, Piper VITS, sanoTTS, Vocetta2, CosyVoice3, VoxCPM2, Voxtral, ... (24) |
| Language models | `text2text` | [Gemma 3 270M IT](https://huggingface.co/loom-ai-org/gemma-3-270m-it-loom) | Qwen3 0.6B, SmolLM2 360M, LFM2 350M, FLAN-T5 small |
| Voice activity, diarization | `speech2class` | [Silero VAD](https://huggingface.co/loom-ai-org/silero-vad-loom) (4 MB) | MarbleNet VAD, pyannote segmentation |
| Language id | `speech2class` | [ECAPA VoxLingua107](https://huggingface.co/loom-ai-org/ecapa-voxlingua107-loom) (107 languages) | |
| Speaker and wake-word embeddings | `speech2embeddings` | [TitaNet-Large](https://huggingface.co/loom-ai-org/titanet-large-loom) | WakeHuBERT tiny |
| Token classification | `text2class` | [DistilBERT NER](https://huggingface.co/loom-ai-org/distilbert-ner-loom) | Punctuate-All (12 languages) |
| Codec tokens, music | `text2codes` + `codes2speech` | [MusicGen small](https://huggingface.co/loom-ai-org/musicgen-small-loom) + [EnCodec](https://huggingface.co/loom-ai-org/encodec-32khz-loom) | Dia + DAC, Qwen3-TTS, MOSS-TTS |

**The full catalogue, with sizes and languages, is in the
[loom.cpp README](https://github.com/loom-ai-org/loom.cpp#supported-models).** Each model card on the Hub
has a ready-to-run example.

## Performance on a Raspberry Pi

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/loom-ai-org/loom.cpp/main/assets/charts/pi4-vs-onnxruntime-dark.svg">
  <img alt="Raspberry Pi 4: loom against onnxruntime. F32: VITS 1.34x, Pocket TTS 2.06x, Qwen3 1.06x, Moonshine 1.13x, whisper 1.00x, Citrinet 0.65x. Int8: VITS 5.03x, Pocket TTS 3.27x, Qwen3 3.45x, Moonshine 1.75x, whisper 1.17x, Citrinet 9.31x." src="https://raw.githubusercontent.com/loom-ai-org/loom.cpp/main/assets/charts/pi4-vs-onnxruntime-light.svg" width="760">
</picture>

| Raspberry Pi 4, 4 threads | loom F32 | onnxruntime FP32 | loom int8 | onnxruntime int8 |
|---|---|---|---|---|
| VITS Piper en-GB · TTS, 3.3 s of speech | **0.80 s** | 1.07 s | **0.82 s** | 4.12 s |
| Pocket TTS · TTS, 4.8 s of speech¹ | **9.11 s** | 18.77 s | **4.25 s** | 13.88 s |
| Qwen3-0.6B · 24 tokens | **16.6 s** | 17.6 s | **4.15 s** | 14.28 s |
| Moonshine tiny · ASR, 11 s clip | **1.34 s** | 1.50 s | **0.85 s** | 1.49 s |
| whisper-small · ASR, 11 s clip | 20.7 s | 20.6 s | **12.5 s** | 14.7 s |
| Citrinet-256 · ASR, 11 s clip | 0.45 s | **0.29 s**² | **0.46 s** | 4.30 s² |

Distilled Pocket TTS on a Pi 4 synthesises 4.8 s of speech in **0.59 s (2M)**, **1.07 s (7M)** and
**1.44 s (18M)** at int8. The tiny models run faster than real time even on a Pi Zero, and its GPU speeds
every one of them up: on a board booted without `vc4-kms-v3d`, `device="auto"` runs the convolutions and
matrix products on its 12 QPUs and the rest on the CPU.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/loom-ai-org/loom.cpp/main/assets/charts/pi-zero-rtf-dark.svg">
  <img alt="Raspberry Pi Zero W real-time factors, CPU then CPU plus VC4: WakeHuBERT 0.17, 0.12; Silero VAD 0.18, 0.16; sanoTTS heartnano 0.22, 0.19; sanoTTS heart 0.53, 0.29; Citrinet-256 0.65, 0.31; sanoTTS thorsten 0.72, 0.35; sanoTTS amy 2.29, 0.69; Moonshine tiny 1.80, 1.18; Conformer-CTC small 1.96, 1.39; VITS piper 5.85, 1.73." src="https://raw.githubusercontent.com/loom-ai-org/loom.cpp/main/assets/charts/pi-zero-rtf-light.svg" width="760">
</picture>

<sub>Pi Zero: each device's faster precision (F32 or Q8_0_PC), the median of three launches. onnxruntime 1.28.0
(PyPI wheel), the same precision, thread count, inputs and output length on both sides; paired medians of five
launches, cooled to 60 °C before each. ¹ Against the community ONNX export
[KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx), since there is no official one.
² onnxruntime is timed from precomputed mel features; loom's time includes its front end. Every table,
with protocol and provenance: loom.cpp's
[docs/BENCHMARKS.md](https://github.com/loom-ai-org/loom.cpp/blob/main/docs/BENCHMARKS.md).</sub>

## Choosing a device

```python
loom.Model.from_file("model.gguf")                 # the best device available (or $LOOM_DEVICE)
loom.Model.from_file("model.gguf", device="cpu")   # pin the CPU
loom.Model.from_file("model.gguf", device="gpu")   # require a GPU; raises if there is none
loom.devices()                                     # what this install can reach
```

Install a GPU backend with `pip install "loom-py-rt[vulkan]"` (or `[cuda]`, `[metal]`). Anything a
backend cannot run falls back to the CPU on its own.

## Contributing

Clone with `--recursive`, since the engine is a submodule:

```sh
git clone --recursive https://github.com/loom-ai-org/loom-py
cd loom-py && pip install -e .
pytest tests/ci        # the Python layer, no model needed (what CI runs)
pytest tests/gate      # end to end on a real GGUF: LOOM_TEST_MODEL=... (+ LOOM_TEST_MODEL_INPUTS as JSON)
```

Design notes, decisions and open work for all three repos live in
[loom.cpp's docs](https://github.com/loom-ai-org/loom.cpp/tree/main/docs). Packaging and wheels are in
[packaging/README.md](https://github.com/loom-ai-org/loom-py/blob/main/packaging/README.md).

| | |
|---|---|
| [**loom.cpp**](https://github.com/loom-ai-org/loom.cpp) | the engine, vendored here as a submodule |
| [**loom-py**](https://github.com/loom-ai-org/loom-py) | this one |
| [**loom-exporter**](https://github.com/loom-ai-org/loom-exporter) | turns a PyTorch checkpoint into a GGUF this runs |

## Licence

MIT, see [LICENSE](https://github.com/loom-ai-org/loom-py/blob/main/LICENSE). Each model on the Hub keeps the
licence of the checkpoint it was exported from.
