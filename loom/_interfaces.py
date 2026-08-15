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
        """Seconds. Zero when the model declared no sample rate, which is honest: without one there is
        no duration to report, only a count of numbers."""
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    def save(self, path: str) -> None:
        """Write a 16-bit PCM WAV. Deliberately `wave` from the standard library rather than soundfile:
        this is the last step of a synthesis call and should not make a package that hands arrays to C++
        acquire an audio dependency to finish its own sentence."""
        import wave

        if not self.sample_rate:
            raise ValueError(
                "this model declared no sample rate, so the samples cannot be written as audio. "
                "`model.contract['sample_rate']` is 0 -- re-export the model with a current "
                "loom-exporter, or write the raw samples yourself at a rate you know."
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

    def _infer(self, text: str | None = None, *, phonemes: Sequence[int] | None = None,
               tokens: Sequence[int] | None = None, steps: int | None = None,
               seed: int | None = None, language: str | None = None, **driver_inputs) -> Audio:
        """Three ways in, and which ones a given model accepts is a property of the model.

        `text` needs a front end that can encode it -- `contract["text_frontend"] == "vocab"`. The four
        phoneme-input families have none embedded, so for them `text` is not a missing feature of this
        package but a step that happens outside the engine, and the error says so rather than pretending.
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
        if seed is not None:
            inputs["seed"] = float(seed)

        samples = self._model.infer(**inputs)
        if not isinstance(samples, list):
            raise TypeError(
                f"this model's driver returned {type(samples).__name__} rather than a waveform. Its "
                f"declared output kind is audio, so either the export or the driver is wrong -- "
                f"`model.driver_source` documents what `infer` actually returns."
            )
        return Audio(samples=[float(s) for s in samples],
                     sample_rate=int(contract.get("sample_rate") or 0))

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
            return [int(p) for p in phonemes]
        if contract.get("text_frontend") != "vocab":
            alphabet = contract.get("phoneme_alphabet") or "phoneme"
            raise UnsupportedTask(
                f"this model takes {alphabet} ids and embeds no vocabulary to produce them from text, "
                f"so it has no text door yet. Pass phonemes=[...] from your own G2P. A phonemizer "
                f"integration is planned (loom.cpp BACKLOG.md Task #79); until it lands this is a real "
                f"limitation of the model rather than a missing convenience here."
            )
        return self._model.tokenize(text, lang=language)


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
Text2Class = _planned("text2class", "text classification")
Speech2Class = _planned("speech2class", "audio classification, language id, keyword spotting")
Image2Class = _planned("image2class", "image classification")
Text2Embeddings = _planned("text2embeddings", "text embedding")
Speech2Embeddings = _planned("speech2embeddings", "speaker embedding, audio embedding")
Image2Embeddings = _planned("image2embeddings", "image embedding")
Image2Boundingbox = _planned("image2boundingbox", "object detection")
Image2Segmentationmask = _planned("image2segmentationmask", "segmentation")


#: Every interface a `Model` carries, in the order `capabilities` and `repr` report them. The three
#: implemented ones first, because that is the order a reader cares about.
ALL_INTERFACES = (
    Text2Text, Speech2Text, Text2Speech,
    Speech2Speech, Text2Image, Image2Text, Speech2Image, Image2Speech,
    Text2Class, Speech2Class, Image2Class,
    Text2Embeddings, Speech2Embeddings, Image2Embeddings,
    Image2Boundingbox, Image2Segmentationmask,
)
