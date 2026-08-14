"""The Vulkan backend for `loom-py-rt`.

There is nothing to import here and nothing to call. This package exists to put one file --
``libggml-vulkan.so`` -- into a directory named ``loom_rt_vulkan`` on ``sys.path``, which is where
``loom/__init__.py`` looks for it. Installing the package is the entire interface::

    pip install "loom-py-rt[vulkan]"

    import loom
    loom.devices()                     # a Vulkan device now appears in the list
    loom.Model.from_file(path, device="gpu")

``loom.devices()`` is the thing to check if it appears not to have worked: a backend whose driver is
too old, or which finds no supported device, loads without error and registers nothing, and the only
visible symptom otherwise is a model that runs at CPU speed.
"""

__all__: list[str] = []


def backend_dir() -> str:
    """The directory holding this package's ``libggml-vulkan.so``.

    Not needed in normal use -- `loom` discovers this package by scanning ``sys.path`` and never
    imports it. It is here for the cases that fall outside discovery: pointing a non-Python host at
    the library, or setting ``$LOOM_BACKEND_DIR`` for a build tree that has its own copy.
    """
    from pathlib import Path

    return str(Path(__file__).resolve().parent)
