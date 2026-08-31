"""The Metal backend for `loom-py-rt`.

There is nothing to import here and nothing to call. This package exists to put one file --
``libggml-metal.so`` -- into a directory named ``loom_rt_metal`` on ``sys.path``, which is where
``loom/__init__.py`` looks for it. Installing the package is the entire interface::

    pip install "loom-py-rt[metal]"

    import loom
    loom.devices()                     # a Metal device now appears in the list
    loom.Model.from_file(path, device="gpu")

``.so`` AND NOT ``.dylib``, WHICH IS NOT A TYPO. ggml builds every backend as a CMake MODULE, and
Darwin gives module libraries the ``.so`` suffix -- which is exactly what ggml's own loader looks
for, since ``backend_filename_extension()`` has no ``__APPLE__`` case. The two agree; see Epic-08 §4.

**Apple Silicon only.** Metal on an Intel Mac is a GPU generation ggml does not target, and Apple
Intel is itself a one-row platform for this project (Epic-08 §4), so there is no ``x86_64`` wheel
here and ``pip install "loom-py-rt[metal]"`` on one will not resolve. That failure is the intended
behaviour rather than an oversight: an extra that quietly resolves to nothing hands back a
successful install and no Metal.

``loom.devices()`` is the thing to check if it appears not to have worked: a backend that finds no
supported device loads without error and registers nothing, and the only visible symptom otherwise
is a model that runs at CPU speed.
"""

__all__: list[str] = []


def backend_dir() -> str:
    """The directory holding this package's ``libggml-metal.so``.

    Not needed in normal use -- `loom` discovers this package by scanning ``sys.path`` and never
    imports it. It is here for the cases that fall outside discovery: pointing a non-Python host at
    the library, or setting ``$LOOM_BACKEND_DIR`` for a build tree that has its own copy.
    """
    from pathlib import Path

    return str(Path(__file__).resolve().parent)
