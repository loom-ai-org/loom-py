# Accelerator packages

`pip install "loom-py-rt[vulkan]"` installs one extra wheel holding one `libggml-vulkan.so`. The base
wheel does not change, is not rebuilt, and is not republished. This directory is how that works and
what to copy when adding CUDA.

## Why not a wheel per accelerator

That is the obvious shape — it is what PyTorch does with `cu121` — and it does not fit here.

PyPI wheel tags encode architecture and libc but have **no accelerator dimension**, so an accelerator
has to be expressed either as a package-name suffix (a full wheel per accelerator) or as something
loaded at run time. The sizes decide it. Measured, release, stripped:

| | |
|---|---|
| `libloom_engine.so` | 1.2 MB |
| `libggml-base.so` + the CPU plugins | ~1.7 MB |
| a CPU-only install | **≈ 3 MB** |
| `libggml-vulkan.so` | **46.5 MB** — 44 MB of it compiled SPIR-V shaders |
| the same install with Vulkan | ≈ 50 MB |

50 MB sits against PyPI's 100 MB per-file ceiling before anything else is added, and CUDA — whose fat
binaries carry cubins per SM architecture — clears it outright. Multiply by five interpreters and
every platform and the matrix is not merely inelegant, it does not fit.

`GGML_BACKEND_DL` is the alternative: backends become shared libraries ggml discovers at run time, so
**one arch-tagged base wheel serves every accelerator**. Note what the sizes also say — the engine is
3% of a Vulkan install and is byte-identical whether ggml ships one backend or nine. Compiling more
backends in never threatens the leanness this project means by the word (that is about code, and about
per-model complexity living in the exporter); what grows is the artifact, and every byte of the growth
is somebody else's precompiled kernels.

## The shape

```
loom-py-rt                     base wheel, arch-tagged     loom/           _loom.so, libloom_engine.so,
                                                                           libggml-base.so, libggml-cpu-*.so
loom-py-rt-vulkan              backend wheel, arch-tagged  loom_rt_vulkan/ libggml-vulkan.so
```

`loom/__init__.py` scans `sys.path` for `loom_rt_*` directories at import and hands each to the engine,
which passes them to `ggml_backend_load_all_from_path`. Nothing is imported and nothing is registered
in the base package per backend, so a new accelerator needs no change to `loom-py-rt` at all.

Discovery order is `$LOOM_BACKEND_DIR`, then the package's own directory, then the `loom_rt_*`
packages — the explicit override first, then what the installation can work out for itself.

## Adding a backend

Copy `rt-vulkan/`, change four things, and build it next to the base wheel:

1. `CMakeLists.txt` — the three arguments to `loom_rt_backend_package` (`cuda`, `loom_rt_cuda`,
   `GGML_CUDA`).
2. `pyproject.toml` — package name, `wheel.packages`, and `[tool.cibuildwheel].archs` (see the table
   below).
3. `loom_rt_<backend>/__init__.py` — the docstring; there is no code in it.
4. `pyproject.toml` in the repo root — a `<backend> = ["loom-py-rt-<backend> == <version>"]` extra.

Most backends are worth building for one or two architectures, so the real matrix is far sparser than
the cross product — roughly nine wheels, not backends times platforms:

| backend | architectures |
|---|---|
| Vulkan | x86-64, aarch64 — the broadest |
| CUDA | x86-64, aarch64 (Jetson/Grace) |
| Metal | arm64 macOS only |
| Hexagon / QNN, RKNPU2 | aarch64 |
| OpenVINO | x86-64 |

**Metal is not CoreML.** Metal is the GPU and is an ordinary in-tree ggml backend; the Neural Engine
means CoreML, which no ggml backend targets. CoreML, RKNPU2 and `ggml-qnn` are all out of tree and
would mean vendoring a backend or carrying a ggml fork — check the licence before any of them, since
this project is MIT and has already turned a dependency down over exactly that.

## Two things that are not negotiable

**The `==` pin.** A backend `.so` links `libggml-base.so.0`, and ggml offers no ABI guarantee across
versions, so any base release that bumps the ggml pin invalidates every backend wheel published before
it. A compatible-release range would silently pair mismatched libraries. The build side of the same
agreement is `cmake/GgmlPin.cmake` in the engine checkout, which both builds read so there is no second
copy of the revision to drift.

**An extra that cannot resolve should fail.** `pip install "loom-py-rt[metal]"` on Linux fails, and
that is intended. The alternative — an environment marker, so the extra quietly resolves to nothing off
macOS — hands back a successful install and no Metal, and the user finds out from an unexplained
performance number much later. It is the same call `Device::open("gpu")` makes in raising rather than
falling back to the CPU. For the same reason there is no `[all]` extra; `[vulkan,cuda]` together is
fine, because the registry picks at run time.

## Checking it worked

```python
import loom
loom.devices()
```

A backend whose driver is too old, or that finds no supported device, loads without error and
registers nothing. `loom.devices()` is where that shows up — otherwise the only symptom is a model
running at CPU speed.
