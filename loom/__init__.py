"""Python bindings for `loom.cpp <https://github.com/femelo/loom.cpp>`_.

A loom model is a single GGUF that carries its own graph topologies and its own driver script
alongside its weights, so this package needs no per-architecture code: loading a model registers
whatever topologies the file declares, and running one calls the driver the file shipped with. A
model this library has never heard of works the day the exporter can produce it.

    import loom

    model = loom.Model.from_pretrained("femelo/qwen3-asr-0.6b-loom")
    print(model.architecture, model.topologies)
    tokens = model.infer(waveform=audio, audio_samples=len(audio))

`infer` is the driver's own entry point and its arguments are the driver's own: a driver takes numbers
and arrays of numbers, and which ones it takes is a property of the model rather than of this package.
`model.driver_source` prints the Lua, whose header comment documents its inputs -- that is the
authority, because it is what will actually run.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import _loom
from ._hub import download

__all__ = ["Model", "LoomError", "download", "__version__"]
__version__ = "0.1.0"

LoomError = _loom.LoomError


class Model:
    """A loaded loom GGUF: its topologies, its hyperparameters and its driver.

    Construct with :meth:`from_file` for a local path or :meth:`from_pretrained` for a HuggingFace
    repo. The constructor takes an already-opened low-level handle and is not the interesting door.
    """

    def __init__(self, handle: "_loom.Model", path: Path):
        self._handle = handle
        self._path = path

    # -- construction --------------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "Model":
        """Load a GGUF from disk."""
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"no such model file: {resolved}")
        return cls(_loom.Model(str(resolved)), resolved)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: str | None = None,
        revision: str = "main",
        cache_dir: str | os.PathLike | None = None,
        token: str | None = None,
    ) -> "Model":
        """Download a GGUF from a HuggingFace repo and load it.

        `filename` may be omitted when the repo holds exactly one `.gguf`, which is the shape a
        single-model repo has; when it holds several, the error lists them rather than guessing, since
        picking one for you is how somebody ends up benchmarking a quantisation they did not choose.
        """
        return cls.from_file(download(repo_id, filename=filename, revision=revision,
                                      cache_dir=cache_dir, token=token))

    # -- what the file says about itself -------------------------------------------------------

    @property
    def architecture(self) -> str:
        """The `loom.architecture` this GGUF declares -- what the exporter called it."""
        return self._handle.architecture()

    @property
    def topologies(self) -> list[str]:
        """Every graph topology the file declares, by name."""
        return list(self._handle.topologies())

    @property
    def has_driver(self) -> bool:
        """Whether the file carries a driver script. Without one it can be inspected but not run."""
        return self._handle.has_driver()

    @property
    def driver_source(self) -> str:
        """The embedded Lua, verbatim. Its header comment documents the inputs `infer` accepts."""
        return self._handle.driver_source()

    @property
    def path(self) -> Path:
        return self._path

    def hparam(self, key: str, kind: str = "u32") -> Any:
        """One `loom.*` hyperparameter, by the type the GGUF stored it as.

        The type is explicit because GGUF's is: a key written as u32 and read as f32 is an error the
        engine raises, and guessing here would only move that error somewhere less clear.
        """
        readers = {"u32": self._handle.hparam_u32, "f32": self._handle.hparam_f32,
                   "str": self._handle.hparam_str}
        if kind not in readers:
            raise ValueError(f"unknown hparam kind {kind!r}; expected one of {sorted(readers)}")
        return readers[kind](key)

    # -- running it ----------------------------------------------------------------------------

    def infer(self, **inputs: float | int | Sequence[float]) -> list[float] | float:
        """Call the model's own driver.

        Keyword arguments become the driver's `inputs` table, so which ones are accepted is a property
        of the model -- `waveform` and `audio_samples` for a speech model, `tokens` for a text one.
        Numbers pass through as numbers; anything sequence-like is flattened to an array of floats,
        which is what the bridge marshals.

        Returns whatever the driver returns: a list for a token sequence or a waveform, a float for a
        single value.
        """
        return self.call("infer", inputs)

    def call(self, fn_name: str, inputs: Mapping[str, Any]) -> list[float] | float:
        """`infer` by another name, for a model whose driver exposes more than one entry point."""
        return self._handle.call(fn_name, {k: _as_value(k, v) for k, v in inputs.items()})

    def __repr__(self) -> str:
        return (f"<loom.Model {self.architecture!r} topologies={self.topologies} "
                f"driver={'yes' if self.has_driver else 'no'} path={self._path.name!r}>")


def _as_value(name: str, value: Any) -> float | list[float]:
    """Coerce one driver input to a number or a list of numbers.

    numpy arrays, array.array, lists and tuples all arrive here; `float(x)` on each element is enough
    for every one of them and avoids making numpy a hard dependency of a package whose whole job is to
    hand arrays to C++.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, bool):
        raise TypeError(
            f"input {name!r} is a bool. A driver's inputs are numbers and arrays of numbers, and a "
            f"bool silently becoming 1.0 is the kind of thing that is only noticed in the output."
        )
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"input {name!r} is a string. Tokenisation happens inside the model -- pass the numbers "
            f"its driver expects, which `model.driver_source` documents."
        )
    if isinstance(value, (Mapping, set, frozenset)):
        # Both are iterable, so without this they would take the sequence path below: a dict would be
        # marshalled as its KEYS, and a set as its elements in whatever order it happened to hold them.
        # A driver input is an ordered array of numbers, and an unordered one arriving as a plausible
        # array is a bug that shows up as bad output rather than as an error.
        raise TypeError(
            f"input {name!r} is a {type(value).__name__}, which has no order to marshal. A driver "
            f"input is a number or an ordered sequence of numbers."
        )
    if isinstance(value, Iterable):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError) as error:
            raise TypeError(f"input {name!r} is a sequence with a non-numeric element: {error}") from None
    raise TypeError(f"input {name!r} is a {type(value).__name__}, which is neither a number nor a sequence")
