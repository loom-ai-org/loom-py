"""Python bindings for `loom.cpp <https://github.com/loom-ai-org/loom.cpp>`_.

A loom model is a single GGUF that carries its own graph topologies and its own driver script
alongside its weights, so this package needs no per-architecture code: loading a model registers
whatever topologies the file declares, and running one calls the driver the file shipped with. A
model this library has never heard of works the day the exporter can produce it.

    import loom

    model = loom.Model.from_pretrained("loom-ai-org/lfm2-350m-loom")
    print(model.generate("The capital of France is", max_new_tokens=14))

`generate` is text in and text out: it tokenizes with the vocabulary the GGUF embeds, runs the driver,
and detokenizes the result. `infer` is the layer under it -- the driver's own entry point, whose
arguments are the driver's own, because which inputs a model takes is a property of the model rather
than of this package. `model.driver_source` prints the Lua, whose header comment documents them; that
is the authority, since it is what will actually run.

A model that embeds no vocabulary -- the phoneme-input TTS families take ids a phonemiser produces
outside the engine -- has `model.tokenizer is None` and is driven through `infer` with ids directly.
That is a property of those models, not of TTS: a model that encodes graphemes itself carries its own
table and tokenizes here like any other.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import _loom
from ._hub import download

__all__ = ["Model", "Tokenizer", "LoomError", "devices", "download", "__version__"]


def _register_backend_paths() -> None:
    """Tell the engine where the ggml backend libraries are.

    This wheel is built with `GGML_BACKEND_DL`: every backend is a shared library loaded at run time,
    including the CPU. That is what lets one arch-tagged wheel serve every accelerator -- an
    accelerator is a separate, small `loom-py-rt-<backend>` package rather than a second copy of
    everything -- but it means nothing at all is available until somebody says where to look.

    ggml's own default is the executable's directory and the current directory. Inside an interpreter
    the executable is `python`, so that default finds nothing and the failure is total rather than
    partial: with no backend loaded there is no CPU either, and every device spec fails. Hence this
    runs at import, before any `Model` can call into device selection.

    Three sources, in the order this project uses everywhere -- the explicit override, then what the
    installation can work out for itself:

    * `$LOOM_BACKEND_DIR`, os.pathsep-separated. For a build tree, a vendored layout, or a backend
      built by hand that no package installed.
    * this package's own directory, which is where the base wheel puts libggml-base.so and the
      per-microarchitecture libggml-cpu-*.so.
    * every `loom_rt_*` package on sys.path -- `loom-py-rt-vulkan` installs `loom_rt_vulkan/` holding
      one libggml-vulkan.so, and `pip install loom-py-rt[vulkan]` is the whole of what a user does.

    Directories are only offered, never required to exist: an accelerator package that is not
    installed is the normal case, not an error.
    """
    for entry in os.environ.get("LOOM_BACKEND_DIR", "").split(os.pathsep):
        if entry:
            _loom.add_backend_search_path(entry)

    _loom.add_backend_search_path(str(Path(__file__).resolve().parent))

    # A directory scan rather than an import or an entry-point lookup, deliberately: it costs no
    # module import, it works the same in an editable install, and a backend package that fails to
    # import for an unrelated reason cannot take the base package down with it.
    seen: set[str] = set()
    for root in sys.path:
        if not root:
            root = os.getcwd()
        try:
            candidates = sorted(Path(root).glob("loom_rt_*"))
        except OSError:  # a sys.path entry that is a zip, a missing directory, or unreadable
            continue
        for candidate in candidates:
            resolved = str(candidate.resolve())
            if candidate.is_dir() and resolved not in seen:
                seen.add(resolved)
                _loom.add_backend_search_path(resolved)


_register_backend_paths()


def devices() -> list[dict]:
    """Every ggml device this process can see, in registration order.

    One entry per device with `name` (what `Model(device=...)` accepts -- "CPU", "Vulkan0", "CUDA0"),
    `description`, `is_cpu`, and `memory_free`/`memory_total` for devices that report them.

    Worth calling after installing an accelerator package, because "did it work" is now an
    install-time question rather than a build-time one: a `loom-py-rt-vulkan` that resolved to the
    wrong architecture, or a driver too old for the .so, shows up here as a device that simply is not
    listed -- which is otherwise indistinguishable from a slow CPU run.
    """
    return _loom.devices()

try:
    # Read from the installed distribution's metadata rather than restating the number here, which
    # drifted from pyproject.toml's `version` before (this hardcoded "0.1.0" while the package shipped
    # 1.0.0rc0) with nothing to catch it.
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("loom-py-rt")
except Exception:  # editable/source-tree checkout with no installed distribution to read
    __version__ = "0+unknown"

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
    def from_file(cls, path: str | os.PathLike, device: str = "") -> "Model":
        """Load a GGUF from disk.

        `device` says where it runs: `""` (the default) defers to `$LOOM_DEVICE` and then to
        autodetection, `"cpu"` pins it to the CPU, `"gpu"` demands a GPU or iGPU, `"npu"` demands an
        accelerator with its own memory, and a device name like `"Vulkan0"` names one exactly. Both
        `"gpu"` and `"npu"` raise rather than fall back, and they do not overlap -- `"gpu"` will not
        answer with an accelerator. With no accelerator package installed there is only a CPU to find,
        so the default resolves there and nothing changes.

        The default ranks by what a device IS -- GPU/iGPU, then an accelerator with its own memory,
        then a host-memory accelerator such as BLAS, then the CPU -- rather than by the order ggml
        registered things, which differs between a linked build and the dynamically loaded one this
        wheel ships (see `loom.cpp` BACKLOG.md P4.8b).
        """
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"no such model file: {resolved}")
        return cls(_loom.Model(str(resolved), device), resolved)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: str | None = None,
        revision: str = "main",
        cache_dir: str | os.PathLike | None = None,
        token: str | None = None,
        device: str = "",
    ) -> "Model":
        """Download a GGUF from a HuggingFace repo and load it.

        `filename` may be omitted when the repo holds exactly one `.gguf`, which is the shape a
        single-model repo has; when it holds several, the error lists them rather than guessing, since
        picking one for you is how somebody ends up benchmarking a quantisation they did not choose.

        `device` is passed through to :meth:`from_file`.
        """
        return cls.from_file(download(repo_id, filename=filename, revision=revision,
                                      cache_dir=cache_dir, token=token), device=device)

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
    def device(self) -> str:
        """The ggml device this session resolved to -- `"CPU"`, `"Vulkan0"`, ...

        Worth reading even when you did not choose one: the default is "decide for me".
        """
        return self._handle.device_name()

    @property
    def device_description(self) -> str:
        """That device in words, e.g. `"AMD Radeon Vega 3 Graphics (RADV RAVEN2)"`."""
        return self._handle.device_description()

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

    # -- text in, text out ---------------------------------------------------------------------

    @property
    def tokenizer(self) -> "Tokenizer | None":
        """The vocabulary this GGUF embeds, or None if it carries none.

        None is the honest answer for the phoneme-input TTS families (Matcha, VITS, Kokoro,
        StyleTTS2): they consume phoneme ids that a phonemiser produces outside the engine, so there
        is no vocabulary in the file to encode text with. It is NOT true of every TTS model -- one
        that encodes graphemes directly, such as Supertonic, carries its own table and reads back
        here like any other vocabulary. See :meth:`generate` for what that means in practice.
        """
        return Tokenizer(self._handle) if self._handle.has_tokenizer() else None

    def tokenize(self, text: str, lang: str | None = None) -> list[int]:
        """Text to token ids, using the model's own embedded vocabulary.

        `lang` is accepted only by a vocabulary that tags its input by language (Supertonic wraps
        text in `<lang>...</lang>` before looking codepoints up); passing one to any other family is
        an error rather than a silently dropped argument. Omit it and the model's own declared
        default is used -- `model.tokenizer.default_lang`, which the export writes into the file.
        """
        return list(self._handle.encode(text, "" if lang is None else lang))

    def detokenize(self, ids: Sequence[float | int]) -> str:
        """Token ids back to text. Floats are accepted because floats are what `infer` returns."""
        return self._handle.decode([int(i) for i in ids])

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        eos_token: int | None = None,
        **inputs: float | int | Sequence[float],
    ) -> str:
        """Text in, text out: tokenize, run the driver, detokenize.

        The end-to-end door for any model whose driver consumes token ids and produces them -- a
        causal LM completing a prompt, most obviously.

        **`tokens` is a convention, not a guess.** Every driver the exporter generates accepts
        `tokens` as an alias for its primary input, whatever the traced graph happens to call it
        (`input_ids`, `token_ids`, `audio_signal`); `driver_components.GENERIC_PRIMARY_INPUT` is where
        that is written down. So this passes `tokens`, and any model carrying a vocabulary works --
        including ones this package has never heard of.

        **Two driver shapes, told apart by what the driver returns rather than by knowing the model.**
        A driver whose cross-step state is entirely the KV cache generates internally and hands back
        the whole sequence; one whose state is not -- LFM2's ten ShortConv blocks, say -- exports a
        single forward pass and returns ONE next token, leaving the loop to the host. A list means the
        first, a number means the second, and the host loop below is what `loom_cli` does for exactly
        that case: append the token, feed the grown prompt back.

        `eos_token` defaults to the model's own `tokenizer.ggml.eos_token_id`, so the loop stops where
        the checkpoint says it should. It only applies to the host-loop shape; a driver that generates
        internally carries its own stop condition.

        For a model whose input is audio rather than text -- an ASR model -- there is nothing to
        encode, so use `model.detokenize(model.infer(waveform=...))`: the same two steps with the
        first one absent.
        """
        return self.detokenize(self.generate_ids(
            self.tokenize(prompt), max_new_tokens=max_new_tokens, eos_token=eos_token, **inputs))

    def generate_ids(
        self,
        tokens: Sequence[int],
        max_new_tokens: int = 64,
        eos_token: int | None = None,
        **inputs: float | int | Sequence[float],
    ) -> list[int]:
        """The token ids `generate` produces, without the encode/decode either side.

        Separate because the ids are sometimes the answer -- comparing against a reference decode,
        or feeding a model whose vocabulary this package cannot read.
        """
        if eos_token is None:
            eos_token = self._handle.kv_i32("tokenizer.ggml.eos_token_id", -1)

        first = self.infer(tokens=list(tokens), max_new_tokens=max_new_tokens,
                           eos_token=eos_token, **inputs)
        if isinstance(first, list):
            return [int(i) for i in first]

        # One token per call. The prompt grows and is re-fed, which is what a driver without
        # cross-step state requires and what `loom_cli` does for this model shape.
        running = [int(i) for i in tokens]
        generated = [int(first)]
        running.append(generated[0])
        while len(generated) < max_new_tokens and generated[-1] != eos_token:
            step = self.infer(tokens=running, max_new_tokens=max_new_tokens,
                              eos_token=eos_token, **inputs)
            token = int(step[-1]) if isinstance(step, list) else int(step)
            generated.append(token)
            running.append(token)
        if generated and generated[-1] == eos_token:
            generated.pop()
        return generated

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
        vocab = self._handle.tokenizer_kind() or "none"
        return (f"<loom.Model {self.architecture!r} topologies={self.topologies} "
                f"driver={'yes' if self.has_driver else 'no'} tokenizer={vocab} "
                f"path={self._path.name!r}>")


class Tokenizer:
    """The vocabulary a GGUF embeds, in whichever of the five families it uses.

    Reached as `model.tokenizer`; `model.tokenize` / `model.detokenize` are the same two calls without
    the intermediate object, which is what most code wants. This exists for the cases where the
    vocabulary itself is the question -- checking `kind` before assuming a model is a text model, or
    `size` against a checkpoint's config.
    """

    def __init__(self, handle: "_loom.Model"):
        self._handle = handle

    @property
    def kind(self) -> str:
        """`gpt2` (byte-level BPE), `bert` (WordPiece), `byt5` (byte-level), `supertonic` (grapheme
        codepoints), or a SentencePiece family name such as `llama` or `t5` -- the GGUF's own
        `tokenizer.ggml.model`."""
        return self._handle.tokenizer_kind()

    @property
    def size(self) -> int:
        return self._handle.tokenizer_size()

    @property
    def default_lang(self) -> str:
        """The language tag :meth:`encode` uses when the caller names none, or `""` for a vocabulary
        that has no such concept -- which is every family but `supertonic`."""
        return self._handle.tokenizer_default_lang()

    def encode(self, text: str, lang: str | None = None) -> list[int]:
        """See :meth:`Model.tokenize`, which is this call without the intermediate object."""
        return list(self._handle.encode(text, "" if lang is None else lang))

    def decode(self, ids: Sequence[float | int]) -> str:
        return self._handle.decode([int(i) for i in ids])

    def __repr__(self) -> str:
        lang = f" lang={self.default_lang!r}" if self.default_lang else ""
        return f"<loom.Tokenizer {self.kind!r} size={self.size}{lang}>"


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
