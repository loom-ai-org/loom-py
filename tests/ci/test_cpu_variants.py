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
import pathlib
import platform
import re

import pytest

import loom

PACKAGE_DIR = pathlib.Path(loom.__file__).resolve().parent

# The lowest rung ggml builds for each architecture, and so the one that decides the oldest machine a
# wheel runs on at all. `x64` is plain x86-64 with no ISA extension beyond the baseline; `armv8.0_1`
# is ARMv8.0-A with none of DOTPROD/FP16/SVE/I8MM/SME. Verified against the published 1.0.0rc3
# wheels: the aarch64 one disassembles with zero dotprod, i8mm, fp16-arithmetic, SVE or SME
# instructions in `armv8.0_1`, and every higher variant contains them.
BASELINE_VARIANT = {
    "x86_64": "libggml-cpu-x64.so",
    "aarch64": "libggml-cpu-armv8.0_1.so",
}


def shipped_variants() -> list[str]:
    return sorted(p.name for p in PACKAGE_DIR.glob("libggml-cpu-*.so"))


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


@pytest.mark.skipif(not pathlib.Path("/proc/self/maps").exists(), reason="Linux-only introspection")
def test_the_mapped_variant_is_exactly_one_the_cpu_supports():
    """Selection ran, and left one winner mapped.

    ggml dlopens every candidate to score it and dlcloses the losers (`RTLD_NOW | RTLD_LOCAL`, no
    `RTLD_NODELETE`), so after the registry is populated exactly one variant remains in
    `/proc/self/maps`. Two would mean the losers are being kept alive; zero means selection never
    ran, which on a machine below the lowest shipped rung is precisely the AVX2 defect.
    """
    loom.devices()  # forces the registry to populate, which is what runs the selection

    maps = pathlib.Path("/proc/self/maps").read_text()
    mapped = sorted(set(re.findall(r"libggml-cpu-[\w.]+\.so", maps)))

    if not shipped_variants():
        pytest.skip("single-variant build: nothing to choose between")

    assert len(mapped) == 1, (
        f"expected exactly one CPU variant mapped after selection, got {mapped}. Zero means no "
        f"variant scored above 0 on this CPU -- shipped: {shipped_variants()}"
    )
    assert mapped[0] in shipped_variants(), (
        f"{mapped[0]} was loaded from outside the package directory: {PACKAGE_DIR}"
    )
