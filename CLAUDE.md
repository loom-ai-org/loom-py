# loom-py — orientation

Python bindings. One of **three repos**, all under `github.com/loom-ai-org` and, on a dev machine,
side by side under one parent directory:

| | |
|---|---|
| `loom.cpp` | the engine — **vendored here as a submodule at `vendor/loom.cpp`**; holds `BACKLOG.md` |
| `loom-exporter` | produces the GGUFs this runs |
| `loom-py` | this repo |

The submodule URL is **relative** (`../loom.cpp`), which resolves to the sibling checkout locally and
to the org on GitHub — one spelling correct in both worlds. Clone with `--recursive`, or the build
fails at `add_subdirectory(vendor/loom.cpp)` on a directory that exists and is empty.

## The one idea

**This package contains no per-architecture code, and should not gain any.** A loom GGUF carries its
own graph topologies and its own driver script, so loading a model means registering whatever
topologies the file declares and handing the engine the Lua it shipped with. A model this package has
never heard of works the day the exporter can produce it. `src/binding.cpp` is deliberately the same
sequence the engine's own end-to-end tests perform, with their hardcoded module lists replaced by
`topology_names()` and the caches attached from `uses_kv_cache()` / `uses_conv_state()`.

If you find yourself special-casing a model name here, the fix belongs in the exporter.

## Build and test

```sh
cmake -B build && cmake --build build -j"$(nproc)"   # writes loom/_loom*.so in place
pytest tests/ci      # 24, no model needed. What CI runs.
pytest tests/gate    # a real GGUF, via LOOM_TEST_MODEL (+ LOOM_TEST_MODEL_INPUTS as JSON)
```

## What the layers own

* **`src/binding.cpp`** — loading, the two caches, tokenizer dispatch across the four vocabulary
  families (taken from `loom.cpp`'s `tools/loom_cli/main.cpp`, including its ordering: `gpt2` is
  dispatched by name because `Vocab::load` throws on a gpt2 file where `BpeVocab::load` returns null).
* **`loom/__init__.py`** — ergonomics and argument coercion. Every rejection there is a value that
  would otherwise have become a plausible wrong number: a bool silently becoming `1.0`, a dict
  marshalled as its keys, a set marshalled in arbitrary order.

**Two driver shapes, told apart by what the driver returns rather than by knowing the model.** One
whose cross-step state is entirely the KV cache generates internally and returns a sequence; one whose
state is not (LFM2's ShortConv blocks) returns a single token and leaves the loop to the host.
`generate_ids` branches on list-vs-number.

**A TTS model has no text door.** Matcha, VITS, Kokoro and StyleTTS2 consume phoneme ids a phonemiser
produces outside the engine, so their GGUFs embed no vocabulary and `model.tokenizer` is `None`. That
is a real limitation, not a missing feature — do not paper over it.
