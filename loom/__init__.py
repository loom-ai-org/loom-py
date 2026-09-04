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
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import _loom
from . import phonemizers
from ._hub import download
from ._interfaces import (ALL_INTERFACES, Audio, Classification, Codes2Speech, Interface,
                          Speech2Text, Text2Class, Text2Codes, Text2Speech, Text2Text, TokenClass,
                          UnsupportedTask)

__all__ = ["Model", "Tokenizer", "Transcription", "Segment", "Audio", "Classification", "TokenClass",
           "Interface", "UnsupportedTask", "Text2Text", "Speech2Text", "Text2Speech", "Text2Class",
           "Text2Codes", "Codes2Speech",
           "phonemizers", "LoomError", "devices", "contract_of", "download", "__version__"]


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


def contract_of(path: str | Path) -> dict:
    """What a loom GGUF declares itself to be, read from the file's metadata alone.

    The same dict as :attr:`Model.contract`, and the same code builds it -- but without loading the
    weights. `Model(path).contract` allocates every tensor on a backend and streams the file into it
    first, which for a large model is gigabytes and seconds spent to read a handful of strings that
    sit in the GGUF header, ahead of the tensor data.

    Use it for the question "is this the file I want" -- which task, which modality pair, which door --
    before committing to loading it::

        if loom.contract_of(path)["interface"] == "text2speech":
            model = loom.Model.from_file(path)

    Everything metadata-shaped is there; nothing that needs a weight is, because no weight was read.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"no such model file: {resolved}")
    return _loom.contract_of(str(resolved))



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
        self._contract = dict(handle.contract())
        # One instance per interface, all of them, always. Most raise -- see `_interfaces.Interface`
        # for why that is better than a method that is absent: "can this model do X" stays a question
        # you can ask, and the answer names what the model actually is.
        for cls in ALL_INTERFACES:
            setattr(self, cls.name, cls(self))

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
        """The `loom.architecture` this GGUF declares -- what the exporter called it.

        A per-MODEL name, and deliberately not what anything here dispatches on: code keyed to it would
        be a table of architecture names living in this package, which is the one thing loom-py is not
        allowed to grow. :attr:`task` and :attr:`contract` are what the high-level doors read.
        """
        return self._handle.architecture()

    @property
    def contract(self) -> dict:
        """What this file says it IS: its task, the modality pair it maps between, and the facts a host
        needs to call it -- sample rate, fixed clip length, whether it can encode text itself.

        `contract["declared"]` is False for a GGUF exported before models stated any of this, which is
        most of them today. That is not an error and not a gap to fill by guessing: the high-level
        interfaces below stay unavailable and the low-level API is unchanged, which is exactly what such
        a file supports.
        """
        return dict(self._contract)

    @property
    def task(self) -> str:
        """The canonical task this model was exported under -- `"text-generation"`,
        `"automatic-speech-recognition"`, `"text-to-speech"` -- or `""` when the file declares none."""
        return self._contract.get("task", "")

    @property
    def capabilities(self) -> tuple:
        """The names of the interfaces this model actually answers to, usually exactly one.

        The same information the interfaces give by raising, without having to call one to find out.
        Empty for a file that declares no contract.
        """
        return tuple(cls.name for cls in ALL_INTERFACES if getattr(self, cls.name).supported)

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

    @property
    def chat_roles(self) -> list[str]:
        """The message roles this checkpoint's chat template can express, or [] if it has none.

        Empty for a base model, and for any GGUF exported before 1.0.0-rc7. **Not always all three**:
        Gemma 3's template folds a system message into the text of the first user turn rather than
        emitting a block for it, so it declares `["user", "assistant"]` and passing it a system message
        is an error rather than a silently dropped argument.
        """
        return list(self._handle.chat_roles())

    def apply_chat_template(
        self,
        messages: Sequence[tuple[str, str]] | Sequence[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        """A conversation as the prompt text this checkpoint was trained on.

        `messages` is `[("user", "..."), ("assistant", "...")]`, or the `{"role": ..., "content": ...}`
        dicts `transformers` uses -- both, because a caller moving code across should not have to
        rewrite the data.

        `add_generation_prompt` appends the opening of the reply the model is being ASKED for, which is
        what turns a transcript into a question. Leave it on unless you are building a training-shaped
        transcript.

        The template itself is data in the GGUF, reduced from the checkpoint's own Jinja at export time
        and verified there against `apply_chat_template`; this package renders no Jinja and carries no
        per-model strings.
        """
        pairs = [(m["role"], m["content"]) if isinstance(m, dict) else (m[0], m[1]) for m in messages]
        return self._handle.apply_chat_template(pairs, bool(add_generation_prompt))

    def chat(
        self,
        messages: str | Sequence[tuple[str, str]] | Sequence[dict],
        max_new_tokens: int = 256,
        **options,
    ) -> str:
        """Ask the model something, in the format it was instruction-tuned on.

        A bare string is the user's turn. Anything else is the whole conversation.

        This is `generate` with the template applied, and applying it is not optional cosmetics: an
        instruction-tuned model given an un-templated prompt behaves like a base model and continues in
        the prompt's own format, which reads exactly like it is repeating you back.

        Sampling and stopping follow the same rules as `generate` -- the checkpoint's own
        `generation_config.json` unless you name something else.
        """
        if isinstance(messages, str):
            messages = [("user", messages)]
        return self.generate(self.apply_chat_template(messages, add_generation_prompt=True),
                             max_new_tokens=max_new_tokens, **options)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        eos_token: int | None = None,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        **inputs: float | int | Sequence[float],
    ) -> str:
        """Text in, text out: tokenize, run the driver, detokenize.

        The same call as ``model.text2text.infer(prompt)``, which is the door this package now names
        every task's end-to-end entry by. Kept because it shipped -- it is public API as of 1.0.0-rc3
        and removing it would break installed code for a rename -- and because for the one task it was
        written for it reads better than the general form.

        The LOOP is the engine's (`loom::text::generate`), not this package's. A Python one lived here
        and was correct, and was still a second copy of a per-task loop: `loom_cli`'s differed in three
        ways -- it ran to the token ceiling with no end-of-sequence stop, took the first element of a
        list return where the new token is the last, and silently rewrote any id >= 65536 to 0. Nothing
        but coincidence kept the two in step, and nothing would have caught them diverging further.

        `eos_token` defaults to the model's own stop SET -- `tokenizer.ggml.eos_token_ids`, which for an
        instruction-tuned checkpoint holds both its base end-of-text and the id a chat turn ends on --
        so generation stops where the checkpoint says it should. Naming one replaces the set rather than
        joining it: "stop here" is an instruction.

        `temperature`/`top_k`/`top_p` default to what the checkpoint's own `generation_config.json`
        declared, which for most models is greedy. `seed` makes a sampled generation reproducible.

        Extra keyword arguments are forwarded to the driver verbatim, for a model whose `infer` takes
        more than tokens.
        """
        return self.detokenize(self.generate_ids(
            self.tokenize(prompt), max_new_tokens=max_new_tokens, eos_token=eos_token,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=seed, **inputs))

    def generate_ids(
        self,
        tokens: Sequence[int],
        max_new_tokens: int = 64,
        eos_token: int | None = None,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        **inputs: float | int | Sequence[float],
    ) -> list[int]:
        """The token ids `generate` produces, without the encode/decode either side.

        Separate because the ids are sometimes the answer -- comparing against a reference decode, or
        feeding a model whose vocabulary this package cannot read.
        """
        # -2 is "ask the file", which is how the engine distinguishes a caller who named no stop token
        # from one who passed -1 to mean "do not stop early" -- the generated drivers' own reading of a
        # negative eos, and a distinction a single sentinel could not carry.
        return list(self._handle.generate(
            [int(t) for t in tokens], int(max_new_tokens),
            -2 if eos_token is None else int(eos_token),
            {k: _as_value(k, v) for k, v in inputs.items()},
            # None all the way down: unset means "decode the way this checkpoint says to", and a
            # checkpoint that says nothing decodes greedily. A default filled in here would silently
            # overrule every file for every caller who named nothing.
            None if temperature is None else float(temperature),
            None if top_k is None else int(top_k),
            None if top_p is None else float(top_p),
            None if seed is None else int(seed)))

    # -- running it ----------------------------------------------------------------------------

    def infer(self, **inputs: float | int | Sequence[float]) -> list[float] | float:
        """Call the model's own driver.

        Keyword arguments become the driver's `inputs` table, so which ones are accepted is a property
        of the model -- `waveform` and `audio_samples` for a speech model, `tokens` for a text one.
        Numbers pass through as numbers; anything sequence-like is flattened to an array of floats,
        which is what the bridge marshals.

        Returns whatever the driver returns: a list for a token sequence or a waveform, a float for a
        single value.

        This is the RAW call and stays raw: one driver invocation, your inputs, no interpretation. For
        speech, `transcribe` is almost certainly what you want -- a model whose graph is built at one
        clip length (Whisper) needs its audio windowed and its decode arguments supplied, and doing
        that yourself is the thing `transcribe` exists to save you.
        """
        return self.call("infer", inputs)

    def transcribe(
        self,
        waveform: Sequence[float],
        *,
        language: str | None = None,
        task: str | None = None,
        timestamps: bool = False,
        condition_on_previous: bool = True,
    ) -> "Transcription":
        """Transcribe a waveform, with everything a long file needs.

        This is the engine's own transcription loop -- the same one `loom_cli` runs, not a reduced
        version of it. Audio for a model whose graph is built at one clip length is windowed and
        zero-padded; each window is decoded with the driver's early stop armed; the output is split
        into timestamped segments; and **the next window starts where the model closed its last
        segment** rather than a fixed stride on, so an utterance straddling a window edge is
        re-decoded whole instead of arriving as two fragments.

        `waveform` is mono floats in [-1, 1] at the rate the model expects -- 16 kHz for every ASR
        family exported so far.

        `language` and `task` are NAMES -- `"en"`, `"translate"` -- not token ids. The engine resolves
        them against the vocabulary the GGUF embeds, which is the only thing that can: `<|en|>` is a
        vocabulary entry, and a caller has no way to look up its id. Leaving them None omits the
        argument, which is how a driver is told to detect or fall back to its own default rather than
        being handed one; naming something this model does not have raises, rather than quietly
        transcribing as if you had not asked.

        Returns a `Transcription`: `.text` for the joined transcript, `.segments` for timed spans.
        """
        options: dict[str, Any] = {"timestamps": timestamps,
                                   "condition_on_previous": condition_on_previous}
        if language is not None:
            options["language"] = str(language)
        if task is not None:
            options["task"] = str(task)
        raw = self._handle.transcribe([float(x) for x in waveform], options)
        # An argument this file has nothing to select with was IGNORED rather than refused -- `language`
        # on a monolingual checkpoint names exactly what it was always going to do. A warning rather
        # than an exception, because the call is correct and its result is what was asked for; a
        # warning rather than silence, because the caller clearly believed the argument did something.
        # A request the model cannot SERVE (a language a multilingual file lacks, `translate` on a file
        # with no task tokens) still raises from the engine.
        for message in raw.get("warnings", ()):
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return Transcription(
            text=raw["text"],
            segments=[Segment(**s) for s in raw["segments"]],
            windows=int(raw["windows"]),
            timestamped=bool(raw["timestamped"]),
        )

    def classify(self, tokens: Sequence[int], *, strip_special: bool = True,
                 **driver_inputs: Any) -> "Classification":
        """Label already-encoded tokens, one declared class each.

        The ids door, matching `generate_ids` beside `generate`: `model.text2class.infer("...")`
        encodes for you and lands here. What this adds over `infer` is the file's own label NAMES and
        the decision about the framing tokens -- both of which are the engine's
        (`loom::text::classify`), so the two front ends cannot disagree about either.

        Returns a `Classification`: the labelled tokens, and the label set they were chosen from.
        """
        ids = [int(t) for t in tokens]
        raw = self._handle.classify(ids, bool(strip_special),
                                    {k: _as_value(k, v) for k, v in driver_inputs.items()})
        return Classification(
            tokens=[TokenClass(token=int(entry["token"]), label_id=int(entry["label_id"]),
                                label=entry["label"], piece=self.detokenize([entry["token"]]))
                    for entry in raw],
            labels=list(self._contract.get("labels") or []),
        )

    def call(self, fn_name: str, inputs: Mapping[str, Any]) -> list[float] | float:
        """`infer` by another name, for a model whose driver exposes more than one entry point."""
        return self._handle.call(fn_name, {k: _as_value(k, v) for k, v in inputs.items()})

    def __repr__(self) -> str:
        vocab = self._handle.tokenizer_kind() or "none"
        return (f"<loom.Model {self.architecture!r} topologies={self.topologies} "
                f"driver={'yes' if self.has_driver else 'no'} tokenizer={vocab} "
                f"path={self._path.name!r}>")


@dataclass(frozen=True)
class Segment:
    """One span of a transcript, in whole-file seconds.

    `closed` is False for text the model had not finished when a window ran out: real transcript, but
    its `end` is the window edge rather than a boundary the model chose -- which matters if you are
    using these times for anything but display.
    """
    start: float
    end: float
    text: str
    closed: bool


@dataclass(frozen=True)
class Transcription:
    """What `Model.transcribe` returns.

    `text` is the segments joined, which is what most callers want. `segments` is there because
    discarding the times would be throwing away something the model computed -- subtitles and seeking
    need them, and re-deriving them is impossible after the fact.

    `timestamped` is False when the model exposes no timestamp tokens at all: the segments are then
    window slices rather than boundaries the model chose, and the long-form seek degrades to fixed
    cuts. Worth checking before trusting the times.
    """
    text: str
    segments: list[Segment]
    windows: int
    timestamped: bool


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
