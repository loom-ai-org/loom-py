#!/usr/bin/env python3
"""Assemble a self-contained source directory for one backend package.

WHY THIS EXISTS. `cibuildwheel` mounts only the package directory into its build container as
`/project`, but a backend package in this repo deliberately reaches OUTSIDE itself for two things:
`packaging/common/BackendPackage.cmake`, which is shared, and the engine submodule's
`cmake/GgmlPin.cmake`, which is where the ggml revision lives. Reaching for the pin rather than
copying it is the mechanism that stops a backend wheel being built against a ggml its base wheel
never saw -- so the fix is not to duplicate it in the repo, but to assemble a complete tree at build
time, from the submodule, that a container can see all of.

    python packaging/stage.py cuda /tmp/stage-cuda
    cibuildwheel --platform linux --archs x86_64 --output-dir dist /tmp/stage-cuda

The staged layout puts the shared and engine cmake modules side by side in `cmake/`, which is the
arrangement `BackendPackage.cmake` looks for first.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def stage(backend: str, dest: Path) -> None:
    package = REPO / "packaging" / f"rt-{backend}"
    if not package.is_dir():
        sys.exit(f"no such backend package: {package}")

    engine_cmake = REPO / "vendor" / "loom.cpp" / "cmake"
    if not (engine_cmake / "GgmlPin.cmake").is_file():
        sys.exit(
            f"the engine submodule is not checked out at {REPO / 'vendor' / 'loom.cpp'} -- "
            "run `git submodule update --init --recursive`. The ggml revision comes from there and "
            "is deliberately not copied into this repo."
        )

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(package, dest)

    # The shared helper and the engine's own modules land together, because BackendPackage.cmake
    # resolves the engine cmake directory as "next to me" when it finds GgmlPin.cmake there.
    cmake_dir = dest / "cmake"
    cmake_dir.mkdir(exist_ok=True)
    shutil.copy2(REPO / "packaging" / "common" / "BackendPackage.cmake", cmake_dir)
    for module in engine_cmake.glob("*.cmake"):
        shutil.copy2(module, cmake_dir)

    pin = (engine_cmake / "GgmlPin.cmake").read_text()
    tag = next((l for l in pin.splitlines() if "LOOM_GGML_TAG" in l and not l.startswith("#")), "?")
    print(f"staged rt-{backend} -> {dest}")
    print(f"  ggml pin taken from the submodule: {tag.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backend", help="backend name, e.g. cuda or vulkan")
    ap.add_argument("dest", type=Path, help="directory to assemble into (replaced if it exists)")
    args = ap.parse_args()
    stage(args.backend, args.dest.resolve())


if __name__ == "__main__":
    main()
