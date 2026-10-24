# loom-py guide

The longer explanation of each task door, what it returns and why, and the low-level layer under them.
The [README](../README.md) has the short version. Moved from the README on 2026-10-24, unchanged.

## Two APIs, and which one you want

**The high-level API is one door per task, named for the modality pair it maps between.** Every model
carries all of them; the ones it does not answer to say so when called, naming what it actually is.

```python
model.text2text.infer("The capital of France is")        # -> str, a raw completion
model.text2text.chat("Who discovered Brazil?")           # -> str, asked inside a turn
model.speech2text.infer(waveform, language="en")         # -> Transcription
model.text2speech.infer("hello world")                   # -> Audio
model.text2class.infer("Wolfgang lives in Berlin")       # -> Classification, a label per token
model.text2codes.infer("[S1] Hello world.")              # -> codec tokens, frame-major
model.codes2speech.infer(codes)                          # -> Audio, the other half of that pair
model.speech2class.infer(waveform)                       # -> AudioClasses, per clip or per frame
model.speech2embeddings.infer(waveform)                  # -> list[float] per clip, or FrameEmbeddings per frame
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

`text2class` returns the labelled pieces beside the label set they came from, because neither is
recoverable from the other — a WordPiece encode splits words, so joining the pieces back up is a rule
only you can make:

```python
r = model.text2class.infer("Wolfgang lives in Berlin")
r.labels                      # ['O', 'B-MISC', 'I-MISC', 'B-PER', ...] — every class the model has
[(t.piece, t.label) for t in r]
# [('Wolfgang', 'B-PER'), ('lives', 'O'), ('in', 'O'), ('Berlin', 'B-LOC')]
```

The framing tokens the encode added ([CLS]/[SEP]) are dropped for you, on the ids the file declares
rather than on their spelling; `strip_special=False` keeps them if you want the raw alignment.

`speech2class` returns the model's own distribution, not a decision — a VAD is thresholded and
smoothed differently by every consumer, and a language id is read top-k — with the time each frame
row starts at, read off the file:

```python
r = vad.speech2class.infer(audio)            # one row per 20 ms frame
r.granularity, r.labels                      # 'frame', ['non_speech', 'speech']
speech = r.probability("speech")             # the speech curve
r.times[:3]                                  # [0.0, 0.02, 0.04] — seconds, frame starts
lid.speech2class.infer(audio).top(3)         # [('th: Thai', 0.988), ('lo: Lao', 0.012), ...]
```

`speech2embeddings` returns the model's vector unnormalised; comparing two (cosine, for a speaker
embedding) and choosing a threshold are yours. A frame-level embedder (WakeHuBERT) returns a
`FrameEmbeddings` instead: `rows` (one vector per frame), `dim`, and `times`, each row's start in seconds.

**A model that speaks through a codec is two files, and the codes are what joins them.** An
autoregressive codec LM emits discrete tokens, not audio; a neural codec turns those tokens into a
waveform. They stay separate because one codec serves many LMs, and because the codes are worth
having on their own — cache them, stream them, decode them at a different rate:

```python
codes = dia.text2codes.infer("[S1] Hello world.")   # -> [[568, 778, ...], ...] frame-major rows
audio = dac.codes2speech.infer(codes)              # -> Audio
```

Nothing goes between the two calls. Both files declare `loom.codec.n_codebooks`, so a pair that does
not fit says so instead of producing audio of the wrong duration:

```python
dia.hparam("codec.n_codebooks", "u32") == dac.hparam("codec.n_codebooks", "u32")
```

`text2codes` counts `max_new_tokens` in **audio frames**, not decoder rows — the two differ by the
model's delay pattern, which is an artefact of how codebooks are written rather than anything you
asked for.

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

*Written when five TTS models were published. Today the phoneme-input ones are Kokoro, Piper VITS,
sanoTTS, Vocetta2, Matcha and StyleTTS2; the rest encode text themselves. Each model card says which.*

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
ceiling. See [`packaging/README.md`](../packaging/README.md).

```python
loom.devices()   # [{'name': 'Vulkan0', 'description': 'AMD Radeon Vega 3 Graphics (RADV RAVEN2)', ...}]
```

Worth calling after installing one, because a backend whose driver is too old — or which finds no
supported device — loads without error and registers nothing, and the only other symptom is a model
running at CPU speed. Note that with this build **every** backend is loaded at run time, the CPU
included, so an empty device list means no backend library was found at all rather than no accelerator.

Which ops fall back to the CPU, and why some always will, is documented in
[loom.cpp's build notes](https://github.com/loom-ai-org/loom.cpp/blob/main/docs/BUILDING.md#running-on-a-gpu).

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

## Model families, door by door

### Language models

These are the models `text2text` answers to; `model.generate(...)` is the same call under the name it
shipped with.

**`infer` and `chat` are different questions.** `infer` CONTINUES a prompt, which is what a base model
does; `chat` puts the prompt inside a turn using the checkpoint's own chat template and asks for a
reply. An instruction-tuned model handed a raw prompt behaves like a base model and continues in the
prompt's own format, which reads exactly like it is repeating you back — so use `chat` for the
instruction-tuned rows above, and `infer` for the base ones. `model.chat_roles` says which a file is:
empty means it carries no template, and Gemma 3's is `['user', 'assistant']` because its template folds
a system message into the first user turn rather than emitting a block for it.

Decoding follows the checkpoint's own `generation_config.json` — Gemma 3 ships `top_k 64, top_p 0.95`
and is sampled; everything else here is greedy. Name `temperature`, `top_k`, `top_p` or `seed` to
override it.

### Text to text, through an encoder-decoder

The same `model.text2text.infer(...)` door as the models above — a host asking for text and getting
text back should not have to know whether one stack or two produced it — but the advice about `chat`
does not apply. This one is instruction-tuned and carries no chat template: the instruction is part of
the text, as in `"translate English to German: ..."` or `"Answer the following question. ..."`.
`model.chat_roles` is empty, which is the file saying so.

### Speech recognition

`model.speech2text.infer(waveform, language="en")` — the mel frontend is inside the graph, so a raw
waveform is the input, and long audio is windowed and seeked for you.

### Speech synthesis

All five take text through `model.text2speech.infer(...)`; the four phoneme-input ones need the
`[phonemes]` extra for the G2P step (see above), and every one of them accepts `phonemes=` or `tokens=`
without it. Kokoro and Supertonic ship a default voice, so a published file speaks on its own.
`model.driver_source` is the authority on what each driver accepts.

### Token classification

`model.text2class.infer("My name is Wolfgang and I live in Berlin")` — one label per token, and the
labels come back beside the PIECES the tokenizer produced rather than beside your words, since a
vocabulary splits some of them. `result.labels` is every class the checkpoint can choose between.

Both answer through that one door and what differs is entirely what a label MEANS. The NER row's
classes are entity tags, so the labels are read as SPANS (`B-PER I-PER` is one person). The
punctuation row's are the mark that follows the token across twelve languages, so they are read back
into a SENTENCE — feed it text with the punctuation already removed, and take each word's mark from
its LAST piece.

### Text to codec tokens, and the codec that decodes them

These two **compose**, and that is the point of the pair: `model.text2codes.infer(...)` turns a
sentence into codec tokens, and `model.codes2speech.infer(...)` on the codec file turns those into a
waveform. Neither is a speech model on its own — the first emits integers and the second takes them —
which is why they declare `audio_codes` rather than `audio` on the side where they meet.
