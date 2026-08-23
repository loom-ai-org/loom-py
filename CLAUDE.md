# loom-py — orientation

Python bindings. One of **three repos**, all under `github.com/loom-ai-org` and, on a dev machine,
side by side under one parent directory:

| | |
|---|---|
| `loom.cpp` | the engine — **vendored here as a submodule at `vendor/loom.cpp`**; holds `docs/`, the shared knowledge base |
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
cmake -B build && cmake --build build -j"$(nproc)"   # writes loom/_loom*.so and the .so it needs
pytest tests/ci      # no model needed. What CI runs.
pytest tests/gate    # a real GGUF, via LOOM_TEST_MODEL (+ LOOM_TEST_MODEL_INPUTS as JSON)
```

## The build is SHARED, and every backend is loaded at run time

`GGML_BACKEND_DL`: `_loom.so`, `libloom_engine.so`, `libggml-base.so` and a set of per-microarchitecture
`libggml-cpu-*.so` all ship inside the `loom/` package directory, wired together with `$ORIGIN` RPATH.
This reverses an earlier deliberate static build — `CMakeLists.txt` states both reasons — and it is what
lets **one arch-tagged wheel serve every accelerator**: `pip install "loom-py-rt[vulkan]"` adds a small
`loom_rt_vulkan` package holding one `libggml-vulkan.so`, and nothing in the base wheel changes. See
`packaging/README.md` for the shape and for what to copy when adding CUDA.

**The consequence to keep in mind: with `GGML_BACKEND_DL` there is no CPU either until something is
loaded.** A registry with no backend .so found is empty, so `device="cpu"` and `device="auto"` fail as
surely as `device="gpu"` does — and ggml's own search (executable directory, current directory) never
finds them from inside an interpreter, where the executable is `python`. `loom/__init__.py` registers
the search paths at import for exactly this reason; `tests/ci/test_backend_discovery.py` is what fails
loudly if that breaks. `loom.devices()` is the way to check what actually loaded.

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

**Four TTS models have no text door, and one does.** Matcha, VITS, Kokoro and StyleTTS2 consume phoneme
ids a phonemiser produces outside the engine, so their GGUFs embed no vocabulary and `model.tokenizer`
is `None`. That is a real limitation, not a missing feature — do not paper over it. Supertonic is not
in that group: it encodes graphemes itself, its GGUF carries the codepoint table, and it tokenizes here
like any other vocabulary. The distinction is per-model, so read the file rather than the task.

**An unrecognized `tokenizer.ggml.model` means no tokenizer, never a failed load.** `Tokenizer::load`
dispatches every family by name and returns null for a tag it does not know. It used to end in an
`else` handing the tag to `Vocab::load`, which throws on anything but `llama`/`t5` — from inside the
`Model` constructor, so a GGUF from a newer exporter did not load at all. A tag it *does* know with the
data missing still throws, on purpose: that file is malformed, not merely new.

## The knowledge base

**Documentation for all three repos lives in `loom.cpp/docs/`**, in four tiers: the open-work hub
([`docs/backlog/active-index.md`](https://github.com/loom-ai-org/loom.cpp/blob/main/docs/backlog/active-index.md)),
domain epics (`docs/epics/`), decisions (`docs/adrs/`) and lessons (`docs/retros/`). The submodule at
`vendor/loom.cpp` has all of it on disk.

This repo's domain is **Epic-06** (the high-level API and its hosts) and **Epic-08** (packaging and
release). Its governing decisions are **ADR-013** (one door per task, declared by the file),
**ADR-009** (backends as dynamic libraries and separate packages) and **ADR-011** (three repos).

Code cites items by `P`-number (`P4.10`, `P4.8g`). The numbers did not change when the old 9,000-line
`BACKLOG.md` was split; that file is now a redirect carrying a map to every section's new home. When
you finish something, put its decision in an ADR, its lesson in a retro, and remove it from the hub —
the routing rules are in `loom.cpp/CLAUDE.md`.
