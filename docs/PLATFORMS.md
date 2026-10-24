# loom-py platforms

Which wheel a machine gets, why each floor is where it is, and the two Raspberry Pi routes. The
[README](../README.md) has the summary. Moved from the README on 2026-10-24, unchanged; the packaging
itself is [`packaging/README.md`](../packaging/README.md).

## Supported platforms

CPython 3.10–3.13, on:

| Platform | Wheel tag | Covers |
|---|---|---|
| Linux x86-64, glibc ≥ 2.28 | `manylinux_2_28_x86_64` | anything not already EOL |
| Linux aarch64, glibc ≥ 2.28 | `manylinux_2_28_aarch64` | **Raspberry Pi 4 and 5 on 64-bit Raspberry Pi OS**, Jetson/Grace, Graviton, Ampere |
| macOS on Apple Silicon, ≥ 14.0 | `macosx_14_0_arm64` | M1 through M4, verified on an M1 Pro |
| macOS on Apple Intel, ≥ 14.0 | `macosx_14_0_x86_64` | built and imported in CI — see below |
| Linux ARMv6, glibc ≥ 2.36 | `linux_armv6l` | **Raspberry Pi Zero / Zero W / Pi 1** — a release asset, not a PyPI install; see below |

There is no per-CPU choice to make. The wheel ships every `libggml-cpu-*.so` variant ggml builds for
the architecture and picks one at import by scoring each against the CPU's own feature flags, so the
same file serves the oldest and newest machine on its architecture. `loom.devices()` is how you check
it worked.

**Why macOS 14 and not 11.** It is not an arbitrary floor and it is not the newest thing that
happened to build. ggml's BLAS backend is compiled against **Accelerate's new BLAS interface**, whose
symbols arrive in macOS 13.3 — and that backend is worth **1.80x on whisper-small**, so dropping it
was the worse trade. An 11.0-tagged wheel would install on macOS 11 or 12 and then quietly have no
BLAS, because a backend that fails to `dlopen` is skipped silently. The floor is the honest version
of what the binaries already require — 14 rather than 13.3 only because a wheel tag carries no minor
version above macOS 11, so 13.3 is not a thing a wheel can say.

**Apple Silicon and Apple Intel are not equally supported, and the table means two different things
by "covers".** The arm64 wheel is built, installed and exercised on a real M1 Pro — the `apple_m1`
rung, which is the *lowest* of ggml's three Apple rungs and so the one every arm64 wheel has to
serve. Apple Intel is **CI-only**: it is built and imported by a runner and nobody here owns the
hardware, so treat it as best-effort rather than as parity.

**Metal is a separate package, not part of the macOS wheel** — `pip install "loom-py-rt[metal]"`.
It is only 0.87 MB, so that split is not about size: a GPU outranks the CPU when you let the library
choose a device, and on unified memory that ranking is not reliably right. Measured on an M1 Pro,
whisper-small is **1.76x faster** on Metal and VITS is **1.79x slower** — so which way it goes still
depends on the model, and asking for the package is how you opt into that trade. `loom.devices()`
shows what you got.

**Raspberry Pi.** Every model in the range is covered, by one of two different routes, and which one
you are on is `uname -m`.

`aarch64` — a Pi 3, 4, 5 or Zero 2 W on **64-bit** Raspberry Pi OS — is a plain
`pip install loom-py-rt` and needs nothing else. This is the route to prefer wherever the board can
take it: a Zero 2 W is ARMv8 hardware, and running it on the 32-bit image gives up NEON and every
`__aarch64__` optimisation in the engine for nothing.

`armv6l` — a **Pi Zero, Zero W or Pi 1** — has a wheel too, and it is *not* on PyPI:

```sh
pip install <the linux_armv6l wheel from https://github.com/loom-ai-org/loom-py/releases/latest>
```

PyPI accepts only `manylinux*` and `musllinux*` platform tags for Linux and the manylinux policy's
floor is armv7, so `linux_armv6l` is refused at upload — there is nothing to fix and no index to wait
for. The wheel is a GitHub release asset instead, built on every release in an emulated Raspbian
(`.github/docker/Dockerfile.armv6`) and checked under `QEMU_CPU=arm1176`, which is the Zero's own
core. Verified on a real Pi Zero W, not only under emulation.

**It is slow, and the Pi 4 numbers on this page do not transfer.** An ARM1176 is one ~1 GHz core with
VFPv2 and no SIMD at all, so this rung runs the generic-C kernels: nothing in `libggml-cpu.so` here is
vectorised, because there is nothing to vectorise with. The 512 MB of RAM is the harder limit — a
small TTS model at Q4_0 is the shape that fits; the 0.6B ASR and LM models are not.

`armv7l` — a Pi 2/3/4 on **32-bit** Raspberry Pi OS — builds by the same path but has no published
wheel and no board here to verify one on, so pip finds nothing and falls back to an sdist build.
Install the 64-bit OS instead; the hardware is ARMv8 either way.

Which variant a board loads, verified under QEMU on every release by `raspberry-pi-check` in
`.github/workflows/wheels.yml`:

| Board | Core | Loads | Because |
|---|---|---|---|
| Pi 4 / 400 / CM4 | Cortex-A72, ARMv8.0-A | `armv8.0_1` | no dotprod, no FP16 arithmetic, no SVE — the other seven variants score 0 |
| Pi 5 / CM5 | Cortex-A76, ARMv8.2-A | `armv8.2_2` | dotprod + FP16, no SVE |

Nothing extra is installed on a Pi: the CPU is the backend. `[vulkan]` does resolve on aarch64 and a
Pi 5's V3D exposes Vulkan through Mesa, but no measurement here says that is worth doing — the CPU
path is the supported one.

**Windows has no wheels**, and a source install does not currently work there. macOS does: both
Apple architectures build from a checkout as well as from a wheel — see *Supported platforms* above.
([Epic-08](https://github.com/loom-ai-org/loom.cpp/blob/main/docs/epics/epic-08-packaging-and-release.md))

From a checkout — note `--recursive`, since the engine is a submodule:

```sh
git clone --recursive https://github.com/loom-ai-org/loom-py
cd loom-py && pip install -e .
```

No runtime dependencies. Arrays cross the boundary as plain sequences of floats, so numpy is something
you may use rather than something this package makes you install — `list`, `array.array`, numpy arrays
and torch tensors all work.
