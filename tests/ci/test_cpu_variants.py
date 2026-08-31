"""Which `libggml-cpu-*.so` the wheel ships, and which one this machine actually maps.

`GGML_CPU_ALL_VARIANTS` (CMakeLists.txt) splits the CPU backend into one shared library per
microarchitecture and ggml picks between them at run time: `ggml_backend_load_best` dlopens every
candidate, calls the `ggml_backend_score()` each one exports, keeps the highest scorer and dlcloses
the rest. A variant scores 0 -- meaning "not usable here" rather than "not preferred" -- when the CPU
lacks a feature it was compiled with, which it learns from `getauxval(AT_HWCAP)` on Linux.

**The failure this guards against is silent and is not hypothetical.** Every wheel published before
2026-08-14 required AVX2, because the build shipped no baseline variant: on a CPU below the lowest
one present, the whole selection returns nothing, and with `GGML_BACKEND_DL` "no CPU variant" means
no devices at all rather than a slow fallback. Nothing in the build fails when a baseline goes
missing -- the wheel builds, imports, and works perfectly on the machine that built it.

The same shape decides whether the aarch64 wheel serves a Raspberry Pi. A Pi 4's Cortex-A72 is
ARMv8.0-A with no dotprod, no half-precision arithmetic and no SVE, so **every ARM variant except
`armv8.0_1` scores 0 there** -- that one library is the entire basis of the Pi 4 claim in README.md.
A Pi 5's Cortex-A76 has dotprod and FP16 but no SVE, so it lands on `armv8.2_2`.

Note what these tests can and cannot see. They run on the machine running them, so CI exercises the
runner's own rung -- an `ubuntu-24.04-arm` runner is Neoverse and picks an 8.2-or-higher variant,
never the Pi 4 one. The baseline test below is what covers the rung nobody's CI machine occupies;
`.github/workflows/wheels.yml`'s `raspberry-pi-check` job is what executes it, under QEMU.
"""
import ctypes
import pathlib
import platform
import re
import sys

import pytest

import loom

PACKAGE_DIR = pathlib.Path(loom.__file__).resolve().parent

# The lowest rung ggml builds for each architecture, and so the one that decides the oldest machine a
# wheel runs on at all. `x64` is plain x86-64 with no ISA extension beyond the baseline; `armv8.0_1`
# is ARMv8.0-A with none of DOTPROD/FP16/SVE/I8MM/SME. Verified against the published 1.0.0rc3
# wheels: the aarch64 one disassembles with zero dotprod, i8mm, fp16-arithmetic, SVE or SME
# instructions in `armv8.0_1`, and every higher variant contains them.
#
# `arm64` IS macOS SPELLING THE SAME ISA `aarch64` NAMES, and it gets a different table entry because
# ggml builds a different ladder there: `apple_m1` (DOTPROD) / `apple_m2_m3` (+I8MM) / `apple_m4`
# (+SME), against Linux-ARM's eight rungs. Every Apple Silicon part has dotprod, so the lowest rung
# is not a compromise baseline the way `armv8.0_1` is -- it covers M1 through M4, and an M1 Pro is
# what actually runs it (Epic-08 §4). Apple Intel takes the `x86_64` row above: ggml resolves it to
# `GGML_SYSTEM_ARCH == "x86"`, whose ladder starts at the same true `x64` baseline.
BASELINE_VARIANT = {
    "x86_64": "libggml-cpu-x64.so",
    "aarch64": "libggml-cpu-armv8.0_1.so",
    "arm64": "libggml-cpu-apple_m1.so",
}


def shipped_variants() -> list[str]:
    """The `.so` suffix is deliberate ON MACOS TOO, and it is the whole of blocker 4's answer.

    ggml's DL loader hardcodes `.so` off `_WIN32` (`backend_filename_extension()`, no `__APPLE__`
    case), which read on its own says a macOS build produces `.dylib` backends its own loader will
    never find -- and with `GGML_BACKEND_DL` that is zero devices, silently. It does not, because
    ggml builds each backend as a CMake **MODULE** (`add_library(${backend} MODULE ...)`), and
    Darwin's `CMAKE_SHARED_MODULE_SUFFIX` is `.so` where `CMAKE_SHARED_LIBRARY_SUFFIX` is `.dylib`.
    Loader and build agree. The linked libraries beside them -- libggml-base, libggml,
    libloom_engine -- are SHARED and genuinely are `.dylib`, which is why the wheel's install rule
    has to match both suffixes while this glob matches one.

    So this function passing on macOS is not incidental: it is the regression test for a build that
    starts emitting `.dylib` backends, which would otherwise show up as an empty `loom.devices()`.
    """
    return sorted(p.name for p in PACKAGE_DIR.glob("libggml-cpu-*.so"))


def mapped_variants() -> list[str] | None:
    """The CPU variant libraries still resident after selection, or None where we cannot look.

    Linux reads `/proc/self/maps`. macOS has no such file; the equivalent is dyld's own image list,
    which `_dyld_image_count`/`_dyld_get_image_name` expose from libSystem and which is already
    linked into every process -- so this needs no new dependency. Both see the same thing: ggml
    dlopens every candidate to score it and dlcloses the losers, so the winner is the only one left.
    """
    maps = pathlib.Path("/proc/self/maps")
    if maps.exists():
        return sorted(set(re.findall(r"libggml-cpu-[\w.]+\.so", maps.read_text())))

    if sys.platform == "darwin":
        libc = ctypes.CDLL(None)
        libc._dyld_image_count.restype = ctypes.c_uint32
        libc._dyld_get_image_name.restype = ctypes.c_char_p
        libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        images = [
            libc._dyld_get_image_name(i).decode("utf-8", "replace")
            for i in range(libc._dyld_image_count())
        ]
        return sorted({m for image in images for m in re.findall(r"libggml-cpu-[\w.]+\.so", image)})

    return None


def test_a_baseline_cpu_variant_is_shipped():
    """The oldest supported CPU has something to load.

    This is the AVX2 defect's regression test, and its ARM twin: without the baseline library the
    package is not slow on an older machine, it is empty on one -- `loom.devices()` returns nothing
    and every `device=` spec fails.
    """
    machine = platform.machine()
    baseline = BASELINE_VARIANT.get(machine)
    if baseline is None:
        pytest.skip(f"no baseline recorded for {machine!r}; add one when this arch is built for")

    variants = shipped_variants()
    if not variants:
        # A single-variant build (`GGML_CPU_ALL_VARIANTS` off) ships `libggml-cpu.so` with no suffix
        # and is tuned to whatever configured it. Legitimate for a local build; this project's
        # CMakeLists forces all-variants on, so a wheel never takes this path.
        assert (PACKAGE_DIR / "libggml-cpu.so").exists(), (
            f"no CPU backend library at all in {PACKAGE_DIR} -- with GGML_BACKEND_DL that is a "
            "package with no devices, not one without an accelerator"
        )
        pytest.skip("single-variant build: nothing to choose between")

    assert baseline in variants, (
        f"{baseline} is missing from {PACKAGE_DIR}, so this wheel silently requires whatever the "
        f"lowest shipped variant needs. Shipped: {variants}"
    )


def test_the_mapped_variant_is_exactly_one_the_cpu_supports():
    """Selection ran, and left one winner mapped.

    ggml dlopens every candidate to score it and dlcloses the losers (`RTLD_NOW | RTLD_LOCAL`, no
    `RTLD_NODELETE`), so after the registry is populated exactly one variant remains resident. Two
    would mean the losers are being kept alive; zero means selection never ran, which on a machine
    below the lowest shipped rung is precisely the AVX2 defect.

    This runs on macOS as well as Linux, and there it is load-bearing rather than a bonus: scoring
    on Apple Silicon goes through `sysctlbyname` in ggml's `arch/arm/cpu-feats.cpp` instead of
    `getauxval`, so "does selection actually pick a variant here" is a separate question from the
    one CI answers on Linux, and only a real Apple part can answer it.
    """
    loom.devices()  # forces the registry to populate, which is what runs the selection

    mapped = mapped_variants()
    if mapped is None:
        pytest.skip(f"no way to enumerate loaded libraries on {sys.platform!r}")

    if not shipped_variants():
        pytest.skip("single-variant build: nothing to choose between")

    assert len(mapped) == 1, (
        f"expected exactly one CPU variant mapped after selection, got {mapped}. Zero means no "
        f"variant scored above 0 on this CPU -- shipped: {shipped_variants()}"
    )
    assert mapped[0] in shipped_variants(), (
        f"{mapped[0]} was loaded from outside the package directory: {PACKAGE_DIR}"
    )
