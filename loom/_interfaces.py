"""The task-based interfaces a `Model` carries, one per modality pair.

WHY THE INTERFACES ARE THE MODALITY PAIR. `Text2Speech` is not a category someone invented for this
package -- it is the I/O contract, which is the axis a task IS (`loom_exporter/tasks.py` makes the same
argument for the export side). So the set of interfaces below needs no lookup table mapping tasks to
doors: a GGUF declares `loom.input.kind` and `loom.output.kind`, the pair names an interface, and the
model that answers to it is the model whose file said so. Nothing here knows an architecture, which is
the rule loom-py's CLAUDE.md states and the reason this layer could not exist before the contract did.

EVERY INTERFACE IS ALWAYS PRESENT, AND MOST OF THEM RAISE. `model.text2image` exists on a speech model
and tells you, when called, that this file is speech->text. That is a better failure than an
AttributeError on a method that was never generated: the question "can this model do X" is answerable by
asking, and the answer names what the model actually is. `Model.capabilities` is the same information
without the raise.

WHAT AN INTERFACE OWNS. Only the preprocessing and postprocessing that surround a driver call:
tokenizing on the way in, detokenizing on the way out, an `Audio` wrapper that keeps a sample rate
attached to its samples. The loops themselves -- the LM decode, the long-form transcription seek --
are the ENGINE's, because a loop written here is a loop the CLI does not get and the two drift (see
loom.cpp docs/HIGH-LEVEL-API.md §1, where they had). An interface that finds itself implementing a
decode strategy is in the wrong layer.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence


class UnsupportedTask(NotImplementedError):
    """Raised by an interface this model does not answer to.

    A `NotImplementedError` subclass so `except NotImplementedError` catches it, and its own type so a
    caller probing several interfaces can tell "this model does not do that" from a genuine gap in this
    package.
    """


@dataclass(frozen=True)
class Audio:
    """A waveform with its sample rate attached.

    The rate travels with the samples because it is not recoverable from them and getting it wrong is
    silent: a 24 kHz waveform played at 22.05 kHz is not an error, it is a slightly deep voice. Returning
    a bare list and expecting the caller to remember is the same defect `Transcription` exists to avoid
    one modality over -- discarding something the model told you.
    """
    samples: list[float]
    sample_rate: int

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        """Seconds, or 0 for an Audio built by hand with no rate -- without one there is no duration to
        report, only a count of numbers."""
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    def save(self, path: str) -> None:
        """Write a 16-bit PCM WAV. Deliberately `wave` from the standard library rather than soundfile:
        this is the last step of a synthesis call and should not make a package that hands arrays to C++
        acquire an audio dependency to finish its own sentence."""
        import wave

        if not self.sample_rate:
            raise ValueError(
                "this Audio has no sample rate, so it cannot be written as audio. A synthesised one "
                "always has a rate (the model's, or the warned default); this is an Audio built by "
                "hand with 0."
            )
        clipped = [max(-1.0, min(1.0, float(s))) for s in self.samples]
        frames = b"".join(int(s * 32767.0).to_bytes(2, "little", signed=True) for s in clipped)
        with wave.open(path, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(self.sample_rate)
            out.writeframes(frames)

    def __array__(self, dtype=None):  # numpy interop without importing numpy
        import numpy as np

        array = np.asarray(self.samples, dtype=dtype or np.float32)
        return array


@dataclass(frozen=True)
class TokenClass:
    """One input token and the class the model gave it."""
    token: int
    label_id: int
    label: str
    piece: str

    def __repr__(self) -> str:
        return f"<{self.piece!r} {self.label or self.label_id}>"


@dataclass(frozen=True)
class Classification:
    """What `Text2Class.infer` returns: a label per token, and the label set they came from.

    A result object rather than a bare list, by the same rule `Transcription` and `Audio` follow: the
    pieces the model actually labelled are not recoverable from the labels afterwards, because a
    WordPiece encode splits words and the alignment is between LABELS and PIECES, not words. A caller
    who joins them back into words is making a rule (which of two pieces' labels wins) that this layer
    has no basis to make for them -- so it hands over both halves.
    """
    tokens: list
    labels: list

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    @property
    def text(self) -> str:
        """The labelled pieces joined back into a string, for reading a result at a glance."""
        return " ".join(f"{t.piece}/{t.label or t.label_id}" for t in self.tokens)


class Interface:
    """Base for every X2Y door. Subclasses declare the modality pair they serve and implement `infer`.

    `kinds` is the pair as the FILE spells it, and several input spellings map to one interface: a model
    taking `phoneme_ids` and one taking `text` are both `Text2Speech`, because supplying ids rather than
    a string is how you talk to a model that cannot encode, not a different contract. That fold happens
    in the engine (`ModelContract::interface_name`) so both hosts agree on it.
    """

    #: The interface name, matching `ModelContract::interface_name()` -- e.g. "speech2text".
    name: str = ""
    #: One-line description, used in the not-supported message so it says what the door is for.
    summary: str = ""

    def __init__(self, model: Any):
        self._model = model

    @property
    def supported(self) -> bool:
        """Whether this model's declared contract names this interface."""
        return self._model.contract.get("interface") == self.name

    def infer(self, *args, **kwargs):
        """Run this model end to end. Raises `UnsupportedTask` when this is not that model's door."""
        if not self.supported:
            raise self._unsupported()
        return self._infer(*args, **kwargs)

    def _infer(self, *args, **kwargs):
        raise self._unsupported()

    def _unsupported(self) -> UnsupportedTask:
        contract = self._model.contract
        if not contract.get("declared"):
            # The honest message for the fleet as it stands: this file predates the contract, so nothing
            # can be said about which door it answers without guessing from its architecture -- which is
            # the guessing this whole layer exists to remove.
            return UnsupportedTask(
                f"this GGUF declares no task, so `{self.name}` cannot be offered for it. It was "
                f"exported before models stated their own contract; re-export it with a current "
                f"loom-exporter, or drive it directly with `model.infer(...)`, which is unchanged."
            )
        return UnsupportedTask(
            f"this model is {contract['interface']} ({contract['task']}), not {self.name}. "
            f"`model.{contract['interface']}.infer(...)` is its door; `model.capabilities` lists what "
            f"it answers to."
        )

    def __repr__(self) -> str:
        state = "available" if self.supported else "not this model"
        return f"<loom.{type(self).__name__} {state}>"


class Text2Text(Interface):
    name = "text2text"
    summary = "text in, text out -- a causal LM completing a prompt"

    def _infer(self, prompt: str, max_new_tokens: int = 64, eos_token: int | None = None,
               **driver_inputs) -> str:
        return self._model.generate(prompt, max_new_tokens=max_new_tokens, eos_token=eos_token,
                                    **driver_inputs)

    def chat(self, messages, max_new_tokens: int = 256, **options) -> str:
        """The same door, asked in the format an instruction-tuned checkpoint was trained on.

        Beside `infer` rather than replacing it, because the two are different questions: `infer`
        CONTINUES a prompt, which is what a base model does and a legitimate thing to ask of any model;
        `chat` puts the prompt inside a turn and asks for a reply. A model whose file carries no chat
        template answers the first and raises on the second, which is the honest split -- `chat_roles`
        on the model says which it is.
        """
        return self._model.chat(messages, max_new_tokens=max_new_tokens, **options)

    def ids(self, tokens: Sequence[int], max_new_tokens: int = 64,
            eos_token: int | None = None, **driver_inputs) -> list[int]:
        """The generated ids, without the encode/decode either side -- sometimes the ids are the answer.

        Both this and `infer` route through `Model.generate_ids` rather than calling the extension a
        second way. The loop underneath is the ENGINE's (`loom::text::generate`); adding a second path
        to it here would be the same duplication one layer down, and this file exists partly because of
        what that cost the last time.
        """
        return self._model.generate_ids(tokens, max_new_tokens=max_new_tokens, eos_token=eos_token,
                                        **driver_inputs)


class Speech2Text(Interface):
    name = "speech2text"
    summary = "audio in, transcript out"

    def _infer(self, waveform: Sequence[float], *, language: str | None = None,
               task: str | None = None, timestamps: bool = False,
               condition_on_previous: bool = True):
        return self._model.transcribe(
            waveform, language=language, task=task, timestamps=timestamps,
            condition_on_previous=condition_on_previous)


class Text2Speech(Interface):
    name = "text2speech"
    summary = "text (or phonemes) in, a waveform out"

    def _infer(self, text: str | None = None, *, phonemes: str | Sequence[int] | None = None,
               tokens: Sequence[int] | None = None, steps: int | None = None,
               seed: int = 0, language: str | None = None,
               sample_rate: int = 16000, **driver_inputs) -> Audio:
        """Three ways in, and which ones a given model accepts is a property of the model.

        `text` needs a front end that can encode it: `"vocab"` encodes it directly, `"phonemes"` runs a
        G2P first and encodes that. A model declaring neither has no text door -- that is a step
        happening outside the engine, and the error says so rather than pretending.

        `phonemes` takes either the STRING a G2P produced (encoded here, through the model's own table
        and its own BOS/EOS assembly) or ids already encoded. `tokens` is the third and lowest: ids
        passed through untouched, assembly included, which is what the driver's own header documents.

        **`sample_rate` is the FALLBACK, not an override.** A model that declares its own rate is
        believed, because the export read it off the checkpoint and the caller is guessing; a model that
        declares none is a real gap today (only Supertonic states it) and this is what fills it. The
        16 kHz default is here in the signature rather than in a module constant so it is visible at the
        call site and replaceable per call -- these families run at 22.05, 24 and 44.1 kHz, so a caller
        who knows their model has somewhere to say so.

        The fallback still warns, because it is still a guess: audio at the wrong rate does not fail, it
        plays at the wrong speed, and that is the only signal a caller gets that the number was invented.
        Passing one you know to be right silences nothing -- there is no way for this to tell a good
        guess from a bad one -- so a model whose rate matters should be re-exported declaring it.
        """
        contract = self._model.contract
        ids = self._resolve_ids(text, phonemes, tokens, language, contract)

        inputs: dict[str, Any] = dict(driver_inputs)
        inputs["tokens"] = [float(i) for i in ids]
        # Declared defaults, applied only when the caller named nothing: a sampler step count is a
        # property of the export (`loom.tts.default_steps`), and a host inventing one is how two front
        # ends produce different audio from the same file.
        if steps is not None:
            inputs["n_steps"] = float(steps)
        elif contract.get("default_steps"):
            inputs["n_steps"] = float(contract["default_steps"])
        # Always passed, and 0 by default. Every sampler here needs one, and a driver handed none fails
        # inside Lua rather than telling a caller what to supply; defaulting it also makes the same text
        # produce the same audio twice, which is the behaviour a caller is more likely to want from a
        # library than from a demo. Name one to vary the voice's noise.
        inputs["seed"] = float(seed)

        samples = self._model.infer(**inputs)
        if not isinstance(samples, list):
            raise TypeError(
                f"this model's driver returned {type(samples).__name__} rather than a waveform. Its "
                f"declared output kind is audio, so either the export or the driver is wrong -- "
                f"`model.driver_source` documents what `infer` actually returns."
            )
        declared = int(contract.get("sample_rate") or 0)
        if not declared:
            warnings.warn(
                f"{self._model.path.name} declares no sample rate, so {sample_rate} Hz is used. If that "
                f"is not this model's rate the audio will play at the wrong speed rather than fail -- "
                f"pass sample_rate= if you know it, or re-export the model declaring it.",
                RuntimeWarning, stacklevel=3,
            )
        return Audio(samples=[float(s) for s in samples], sample_rate=declared or int(sample_rate))

    def _resolve_ids(self, text, phonemes, tokens, language, contract) -> Sequence[int]:
        named = [n for n, v in (("text", text), ("phonemes", phonemes), ("tokens", tokens))
                 if v is not None]
        if len(named) != 1:
            raise TypeError(
                f"give exactly one of text=, phonemes= or tokens= (got {named or 'none'}). They are "
                f"three depths of the same input, not alternatives to combine."
            )
        if tokens is not None:
            return [int(t) for t in tokens]
        if phonemes is not None:
            # A STRING goes through the model's own table, ids go straight through. Both spellings are
            # what a caller actually holds: `phonemize()` returns a str and so does every external G2P,
            # while a caller who has already encoded holds ids. Before, `phonemes=` was `[int(p) for p
            # in phonemes]` -- byte-identical to the `tokens=` branch above, so the str form died on
            # `invalid literal for int() with base 10: 'h'` and the two parameters were one parameter
            # under two names. Encoding here rather than telling the caller to call `tokenize` first is
            # also what applies the model's own BOS/EOS assembly, which is exactly the step a
            # bring-your-own-G2P caller has no way to know about (Kokoro wraps with 0 at both ends).
            if isinstance(phonemes, str):
                if self._model.tokenizer is None:
                    raise UnsupportedTask(
                        f"this model embeds no symbol table, so it cannot encode a phoneme string. Pass "
                        f"phonemes=[ids] or tokens=[ids] from your own table, or re-export it with a "
                        f"current loom-exporter, which writes the table its checkpoint already had."
                    )
                return self._model.tokenize(phonemes)
            return [int(p) for p in phonemes]
        frontend = contract.get("text_frontend")
        if frontend == "phonemes":
            # Two steps, and only the second is in the file: G2P is a property of the LANGUAGE and lives
            # outside every GGUF, while the symbol table that turns its output into ids is the
            # checkpoint's own and now travels with it. So this phonemizes here and encodes there, which
            # is why a model with a phoneme vocabulary still needs an installed provider to take text.
            from . import phonemizers

            alphabet = contract.get("phoneme_alphabet") or "ipa"
            if self._model.tokenizer is None:
                raise UnsupportedTask(
                    f"this model takes {alphabet} ids and embeds no symbol table to produce them from, "
                    f"so it has no text door. Pass phonemes=[...] from your own G2P, or re-export it "
                    f"with a current loom-exporter, which writes the table its checkpoint already had."
                )
            spoken = phonemizers.phonemize(
                text, alphabet=alphabet, language=language or (contract.get("languages") or ["en"])[0])
            return self._model.tokenize(spoken)
        if frontend != "vocab":
            raise UnsupportedTask(
                f"this model declares no text front end, so it cannot encode text. Pass tokens=[...] "
                f"or phonemes=[...] directly, which is what `model.driver_source` documents it taking."
            )
        return self._model.tokenize(text, lang=language)


class Text2Class(Interface):
    name = "text2class"
    summary = "text in, one declared class per token out"

    def _infer(self, text: str | None = None, *, tokens: Sequence[int] | None = None,
               strip_special: bool = True, **driver_inputs) -> Classification:
        """Label a sentence, one class per token.

        Two ways in, the same ladder every other door offers: `text` is encoded through the model's own
        vocabulary, `tokens` are ids a caller already holds. There is no third rung here because there
        is no intermediate representation -- a token classifier's input is the tokenizer's output and
        nothing sits between them.

        `strip_special` drops the rows belonging to the framing tokens the encode adds ([CLS]/[SEP] for
        a WordPiece model). It is the ENGINE's decision, made on the ids the file declares rather than
        on their spelling; this parameter is how a caller who wants the raw alignment turns it off.
        """
        if (text is None) == (tokens is None):
            raise TypeError(
                "give exactly one of text= or tokens=. They are two depths of the same input, not "
                "alternatives to combine."
            )
        if text is not None:
            if self._model.tokenizer is None:
                raise UnsupportedTask(
                    "this model embeds no vocabulary, so it cannot encode text. Pass tokens=[ids] "
                    "from your own tokenizer."
                )
            ids = self._model.tokenize(text)
        else:
            ids = [int(t) for t in tokens]
        return self._model.classify(ids, strip_special=strip_special, **driver_inputs)


class Codes2Speech(Interface):
    name = "codes2speech"
    summary = "neural-codec tokens in, a waveform out"

    def _infer(self, codes, *, sample_rate: int = 0, **driver_inputs) -> Audio:
        """Decode codec tokens to audio.

        `codes` is frame-major -- one row per frame, `n_codebooks` wide -- which is the order an
        AR codec-token LM emits in and the layout the export declares. A flat sequence is accepted too
        and is taken to be that same matrix already flattened, because that is what a driver that
        produced it hands over.

        **What this door does NOT do is undo a delay pattern.** An AR LM offsets codebook *k* by *k*
        steps; realigning them is a property of that LM, not of the codec, and a codec asked to guess
        would be guessing about a model it has never seen. Feed it aligned codes.
        """
        rows = self._as_matrix(codes)
        declared = int(self._model.contract.get("sample_rate") or 0)
        if not declared and not sample_rate:
            warnings.warn(
                f"{self._model.path.name} declares no sample rate, so 24000 Hz is assumed. A wrong "
                f"rate does not fail, it plays the audio at the wrong speed -- pass sample_rate= if "
                f"you know it.", RuntimeWarning, stacklevel=3,
            )
        samples = self._model.infer(codes=[float(c) for row in rows for c in row], **driver_inputs)
        if not isinstance(samples, list):
            raise TypeError(
                f"this model's driver returned {type(samples).__name__} rather than a waveform. Its "
                f"declared output kind is audio, so either the export or the driver is wrong."
            )
        return Audio(samples=[float(s) for s in samples],
                     sample_rate=declared or int(sample_rate) or 24000)

    def _as_matrix(self, codes) -> list:
        """`codes` as a list of per-frame rows, whichever of the two shapes the caller holds.

        The width is checked against what the FILE declares rather than inferred: a flat list of the
        wrong length would otherwise be silently reinterpreted as a different number of frames, which
        produces audio of the wrong duration and no error anywhere.
        """
        # An HPARAM, not a contract field: ADR-020 puts it there because it is a number the HOST needs
        # in order to build an input, which is the split `hparams()` already draws. `hparam` raises
        # when the key is absent, and absent means an export too old to state it -- which is a
        # different thing from a model with no codebooks, so it is caught rather than defaulted.
        try:
            width = int(self._model.hparam("codec.n_codebooks", "u32"))
        except Exception:
            width = 0
        rows = list(codes)
        if rows and isinstance(rows[0], (list, tuple)):
            rows = [list(r) for r in rows]
            bad = [i for i, r in enumerate(rows) if width and len(r) != width]
            if bad:
                raise ValueError(
                    f"this model takes {width} codebooks per frame; rows {bad[:5]} have "
                    f"{[len(rows[i]) for i in bad[:5]]}."
                )
            return rows
        if not width:
            raise ValueError(
                "this model declares no `loom.codec.n_codebooks`, so a flat list cannot be split into "
                "frames. Pass a list of per-frame rows, or re-export it with a current loom-exporter."
            )
        if len(rows) % width:
            raise ValueError(
                f"{len(rows)} codes is not a whole number of frames at {width} codebooks per frame. "
                f"Codes are frame-major: all {width} codes for frame 0, then frame 1, and so on."
            )
        return [rows[i:i + width] for i in range(0, len(rows), width)]


class _Planned(Interface):
    """An interface named by the taxonomy with no family exporting to it yet.

    They exist as members rather than being added when the first model arrives, and that is the point:
    the set of doors is a property of the modality taxonomy, not of what happens to be implemented this
    month. `model.image2text.infer(...)` says "no loom model does this yet" instead of AttributeError,
    which is the difference between a roadmap and a typo.
    """

    def _infer(self, *args, **kwargs):
        raise UnsupportedTask(
            f"`{self.name}` is a declared interface with no exported family behind it yet "
            f"({self.summary}). No GGUF can currently declare that contract, so nothing answers here."
        )


def _planned(interface_name: str, summary: str) -> type:
    return type(
        "".join(part.capitalize() for part in interface_name.split("2")).replace("2", "2"),
        (_Planned,),
        {"name": interface_name, "summary": summary},
    )


Speech2Speech = _planned("speech2speech", "voice conversion, speech translation")
Text2Image = _planned("text2image", "image synthesis")
Image2Text = _planned("image2text", "captioning, OCR, VLM prompting")
Speech2Image = _planned("speech2image", "speech-conditioned image synthesis")
Image2Speech = _planned("image2speech", "image-conditioned speech")
Speech2Class = _planned("speech2class", "audio classification, language id, keyword spotting")
Image2Class = _planned("image2class", "image classification")
Text2Embeddings = _planned("text2embeddings", "text embedding")
Speech2Embeddings = _planned("speech2embeddings", "speaker embedding, audio embedding")
Image2Embeddings = _planned("image2embeddings", "image embedding")
Image2Boundingbox = _planned("image2boundingbox", "object detection")
Image2Segmentationmask = _planned("image2segmentationmask", "segmentation")


#: Every interface a `Model` carries, in the order `capabilities` and `repr` report them. The five
#: implemented ones first, because that is the order a reader cares about.
ALL_INTERFACES = (
    Text2Text, Speech2Text, Text2Speech, Text2Class, Codes2Speech,
    Speech2Speech, Text2Image, Image2Text, Speech2Image, Image2Speech,
    Speech2Class, Image2Class,
    Text2Embeddings, Speech2Embeddings, Image2Embeddings,
    Image2Boundingbox, Image2Segmentationmask,
)
