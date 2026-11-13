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

import re
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
    # Interleaved channels: 2 means `samples` runs `L R L R ...`. Travels with the samples for the
    # rate's reason -- a stereo run read as mono plays at half speed with nothing raised.
    channels: int = 1

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        """Seconds, or 0 for an Audio built by hand with no rate -- without one there is no duration to
        report, only a count of numbers."""
        if not self.sample_rate:
            return 0.0
        return len(self.samples) / self.channels / self.sample_rate

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
            # Interleaved samples are already in a WAV data chunk's own frame order.
            out.setnchannels(self.channels)
            out.setsampwidth(2)
            out.setframerate(self.sample_rate)
            out.writeframes(frames)

    def __array__(self, dtype=None):  # numpy interop without importing numpy
        """`[samples]` for mono, `[frames, channels]` for interleaved audio -- soundfile's layout."""
        import numpy as np

        array = np.asarray(self.samples, dtype=dtype or np.float32)
        return array.reshape(-1, self.channels) if self.channels > 1 else array


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


@dataclass(frozen=True)
class AudioClasses:
    """What `Speech2Class.infer` returns: the model's own class distribution, per clip or per frame.

    **Probabilities, not a decision.** A VAD's consumers threshold, smooth and hang over its output
    each in their own way, and a language id is routinely read top-k -- so the answer is the whole
    distribution, row by row, and `best`/`top` are conveniences over it rather than a rule this layer
    makes for you. The cut into rows, the label names and the time each frame row covers are the
    ENGINE's (`loom::audio::classify`), read off the file's contract, so loom_cli and this agree.

    `granularity` is "clip" (one row) or "frame" (one row per encoder frame, starting at
    `frame_offset` seconds and `1 / frame_rate` apart).
    """
    granularity: str
    labels: list
    probabilities: list  # rows of len(labels) probabilities
    frame_rate: float = 0.0
    frame_offset: float = 0.0

    def __len__(self) -> int:
        return len(self.probabilities)

    @property
    def best(self) -> list:
        """The most probable label of each row."""
        return [self.labels[max(range(len(row)), key=row.__getitem__)] for row in self.probabilities]

    @property
    def times(self) -> list:
        """Each row's start in seconds -- all 0.0 for a clip answer, which covers the whole clip."""
        if not self.frame_rate:
            return [0.0] * len(self.probabilities)
        return [self.frame_offset + i / self.frame_rate for i in range(len(self.probabilities))]

    def top(self, k: int = 5, row: int = 0) -> list:
        """`[(label, probability), ...]`, the `k` most probable classes of one row -- the clip's, by
        default."""
        pairs = sorted(zip(self.labels, self.probabilities[row]), key=lambda lp: -lp[1])
        return pairs[:k]

    def probability(self, label: str) -> list:
        """One label's probability in every row -- a VAD's speech curve, for instance."""
        k = self.labels.index(label)
        return [row[k] for row in self.probabilities]


@dataclass(frozen=True)
class FrameEmbeddings:
    """What `Speech2Embeddings.infer` returns for a frame-level embedder: one vector per encoder frame.

    A clip embedder answers with the bare vector, because there is nothing to attach to it; a frame
    answer has the time each row covers, so it gets the object `AudioClasses` already is. The cut into
    rows (by the file's declared `embedding_dim`) and the times are the ENGINE's (`loom::audio::embed`),
    so loom_cli and this agree. Row `i` starts `frame_offset + i / frame_rate` seconds into the clip.

    The values are the model's own, unnormalised; what a detector does with them is the caller's.
    """
    rows: list  # one list of `dim` floats per frame
    dim: int
    frame_rate: float
    frame_offset: float = 0.0

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def times(self) -> list:
        """Each row's start in seconds."""
        if not self.frame_rate:
            return [self.frame_offset] * len(self.rows)
        return [self.frame_offset + i / self.frame_rate for i in range(len(self.rows))]


class FrameStream:
    """A frame classifier's or frame embedder's rows as the audio arrives, from `speech2class.stream()` or
    `speech2embeddings.stream()` (loom.cpp ADR-106).

        vad = model.speech2class.stream()
        for piece in microphone():   # mono floats at contract["sample_rate"], any length
            for t, p in zip(*(u := vad.push(piece)).times, u.probability("speech")):
                ...
        vad.finish()

    Each `push` returns what one `infer` call returns -- an `AudioClasses` or a `FrameEmbeddings` -- for the
    frames that push completed, with `frame_offset` set to the first one's time in the stream, so `.times`
    are stream times. The rows of every call joined are `infer`'s rows over the whole audio: where the
    pieces break never changes them. The cut into rows and their times are the engine's.
    """

    def __init__(self, handle):
        self._handle = handle
        layout = handle.layout()
        self.kind = str(layout["kind"])
        self.labels = list(layout["labels"])
        self.width = int(layout["width"])
        self.frame_rate = float(layout["frame_rate"])
        self.frame_offset = float(layout["frame_offset"])

    def push(self, samples: Sequence[float]) -> "AudioClasses | FrameEmbeddings":
        """Add audio and answer every frame it completes."""
        return self._answer(self._handle.push([float(x) for x in samples]))

    def finish(self) -> "AudioClasses | FrameEmbeddings":
        """End the input: answer what is left. Takes no push afterwards until `reset`."""
        return self._answer(self._handle.finish())

    def reset(self) -> None:
        """Drop everything and start again on the same model."""
        self._handle.reset()

    @property
    def rows(self) -> int:
        """Frames answered so far."""
        return int(self._handle.rows())

    @property
    def finished(self) -> bool:
        return bool(self._handle.finished())

    @property
    def state_bytes(self) -> int:
        """What this stream's state holds, on the device and on the host."""
        return int(self._handle.state_bytes())

    def _answer(self, raw: dict):
        flat, w = raw["values"], self.width
        rows = [list(flat[r * w:(r + 1) * w]) for r in range(int(raw["n_rows"]))]
        offset = self.frame_offset + int(raw["first_row"]) / self.frame_rate
        if self.kind == "class":
            return AudioClasses(granularity="frame", labels=list(self.labels), probabilities=rows,
                                frame_rate=self.frame_rate, frame_offset=offset)
        return FrameEmbeddings(rows=rows, dim=w, frame_rate=self.frame_rate, frame_offset=offset)


class SpeechStream:
    """A TTS's audio handed back a piece at a time, from `text2speech.stream`. Iterate it for one
    :class:`Audio` per piece, in order; a piece that made no sound is skipped.

    `exact` says how it was cut. True: by the model's own driver (loom.cpp ADR-100, ADR-101), and the pieces
    joined are `infer`'s waveform, to within f32 rounding where the driver decodes a frame at a time -- or,
    for a model whose reference streams in a mode of its own (CosyVoice3, ADR-112), `infer` in that mode. False:
    here, a sentence of the text per `infer` call, which differs from one call over the whole text at the
    joins.
    """

    def __init__(self, handle, sample_rate: int, exact: bool):
        self._handle = handle
        self.sample_rate = int(sample_rate)
        self.exact = bool(exact)

    def __iter__(self):
        while not self.finished:
            piece = self.next()
            if piece is not None:
                yield piece

    def next(self) -> Audio | None:
        """The next piece, or None once finished."""
        samples = self._handle.next()
        if len(samples) == 0:
            return None
        return Audio(samples=[float(s) for s in samples], sample_rate=self.sample_rate)

    def reset(self) -> None:
        """Back to the start: the next piece is the first again."""
        self._handle.reset()

    @property
    def finished(self) -> bool:
        return bool(self._handle.finished())

    @property
    def pieces_done(self) -> int:
        return int(self._handle.pieces_done())

    @property
    def pieces_total(self) -> int:
        """How many pieces there are, or 0 when the model does not say up front."""
        return int(self._handle.pieces_total())

    @property
    def seconds(self) -> float:
        """Audio handed back so far."""
        return int(self._handle.samples()) / self.sample_rate

    @property
    def state_bytes(self) -> int:
        """What the engine holds for this stream between pieces."""
        return int(self._handle.state_bytes())


class CodesStream:
    """A text-to-codes model's frames handed back as they are drawn, from `text2codes.stream`. Iterating
    yields each piece's frames as rows, `n_codebooks` wide; a `codes2speech.stream` takes the stream as
    it is."""

    def __init__(self, handle, interface: "Text2Codes"):
        self._handle = handle
        self._interface = interface

    def __iter__(self):
        while not self.finished:
            frames = self.next()
            if frames:
                yield frames

    def next(self) -> list[list[int]]:
        """The next piece's frames; [] once finished."""
        return self._interface._as_frames(self._handle.next())

    def reset(self) -> None:
        self._handle.reset()

    @property
    def finished(self) -> bool:
        return bool(self._handle.finished())

    @property
    def state_bytes(self) -> int:
        return int(self._handle.state_bytes())


class CodecStream:
    """A codec decoding codes as they arrive, from `codes2speech.stream`. `push(codes)` returns their
    `Audio`; `finish()` whatever is left. Iterating it pushes each item of the `frames` it was given."""

    #: What a push grows to when `frames_per_push` is not given (ADR-102).
    GROWN = 8

    def __init__(self, handle, interface: "Codes2Speech", sample_rate: int, channels: int, frames=None,
                 frames_per_push: int | None = None):
        if frames_per_push is not None and frames_per_push < 1:
            raise ValueError(f"frames_per_push is at least one frame, not {frames_per_push}")
        self._handle = handle
        self._interface = interface
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._frames = frames
        self._per_push = frames_per_push

    def _audio(self, samples) -> Audio:
        return Audio(samples=[float(s) for s in samples], sample_rate=self.sample_rate, channels=self.channels)

    def push(self, codes) -> Audio:
        """`codes` -- rows or a flat frame-major run, whole frames -- decoded after everything pushed
        before them."""
        rows = self._interface._as_matrix(codes)
        return self._audio(self._handle.push([int(c) for row in rows for c in row]))

    def finish(self) -> Audio:
        return self._audio(self._handle.finish())

    def reset(self) -> None:
        self._handle.reset()

    def __iter__(self):
        if self._frames is None:
            raise TypeError("this stream was opened without frames to decode: call push() with codes instead")
        pending, pushes = [], 0
        for codes in self._frames:
            pending += self._interface._as_matrix(codes)
            want = self._per_push or min(self.GROWN, 2 ** pushes)
            if len(pending) >= want:
                audio = self.push(pending)
                pending, pushes = [], pushes + 1
                if len(audio.samples):
                    yield audio
        if pending:
            audio = self.push(pending)
            if len(audio.samples):
                yield audio
        tail = self.finish()
        if len(tail.samples):
            yield tail

    @property
    def finished(self) -> bool:
        return bool(self._handle.finished())

    @property
    def seconds(self) -> float:
        return int(self._handle.samples()) / self.channels / self.sample_rate

    @property
    def state_bytes(self) -> int:
        return int(self._handle.state_bytes())


# A sentence ends at a run of terminal marks (with any closing quote or bracket) followed by white space,
# at a CJK terminal mark, or at a line break. Generic on purpose: this is the cut for models whose own
# driver does not cut, and "Dr. Smith" becomes two pieces, which costs a pause and nothing else.
_SENTENCE_END = re.compile(r"[.!?\u2026]+[\"')\]\u201d\u2019]*(?=\s)|[\u3002\uff01\uff1f]+|\n+")


def split_sentences(text: str) -> list[str]:
    """`text` cut into sentences, each stripped; the whole text when nothing cuts it."""
    out, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        piece = text[start:m.end()].strip()
        if piece:
            out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out or [text]


class _SentencePieces:
    """The stream handle for a model whose driver does not stream: one `infer` per piece."""

    def __init__(self, interface: "Text2Speech", pieces: list[dict], options: dict):
        self._interface = interface
        self._pieces = pieces
        self._options = options
        self.reset()

    def next(self) -> list[float]:
        if self.finished():
            return []
        piece = {k: v for k, v in self._pieces[self._done].items() if v is not None}
        inputs = self._interface._inputs(**piece, **self._options)
        samples = self._interface._model.infer(**inputs)
        self._done += 1
        self._samples += len(samples)
        return samples

    def reset(self) -> None:
        self._done = 0
        self._samples = 0

    def finished(self) -> bool:
        return self._done >= len(self._pieces)

    def pieces_done(self) -> int:
        return self._done

    def pieces_total(self) -> int:
        return len(self._pieces)

    def samples(self) -> int:
        return self._samples

    def state_bytes(self) -> int:
        return 0


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

    def stream(self, prompt, max_new_tokens: int = 64, **options):
        """`infer`, handed back as it is drawn (`Model.generate_stream`): iterate it for the text."""
        if not self.supported:
            raise self._unsupported()
        return self._model.generate_stream(prompt, max_new_tokens=max_new_tokens, **options)

    def chat_stream(self, messages, max_new_tokens: int = 256, **options):
        """`chat`, handed back as it is drawn (`Model.chat_stream`)."""
        if not self.supported:
            raise self._unsupported()
        return self._model.chat_stream(messages, max_new_tokens=max_new_tokens, **options)

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
               task: str | None = None, target_language: str | None = None,
               timestamps: bool = False, condition_on_previous: bool = True):
        return self._model.transcribe(
            waveform, language=language, task=task, target_language=target_language,
            timestamps=timestamps, condition_on_previous=condition_on_previous)

    def stream(self, **options):
        """A live transcription (`Model.stream`): push audio as it arrives, read text as it grows.
        `language` is a name, as `transcribe` takes it; `options` go to the driver (`update_seconds` for a
        re-decoding file). Raises `UnsupportedTask` on a
        model that is not speech2text, and `RuntimeError` on one whose file has no live path yet
        (`model.can_stream`)."""
        if not self.supported:
            raise self._unsupported()
        return self._model.stream(**options)


class Speech2Class(Interface):
    name = "speech2class"
    summary = "audio in, a class distribution per clip or per frame out (language id, VAD, segmentation)"

    def _infer(self, waveform: Sequence[float]) -> AudioClasses:
        """Classify a clip: mono floats in [-1, 1] at the model's own `contract["sample_rate"]`.

        One call over the whole clip -- no windowing, which is the model's to want and none of family
        13's do. Returns an `AudioClasses`: per clip for a language id, per frame for a VAD or a
        segmentation model.
        """
        return self._model.classify_audio(waveform)

    def stream(self) -> FrameStream:
        """Classify live (`FrameStream`): push audio as it arrives, read each frame's distribution once its
        audio is in. Per-frame models only -- a clip's one row needs the whole clip. Raises
        `UnsupportedTask` on a model that is not speech2class, and `RuntimeError` on one whose file has no
        live path (`model.can_stream_frames`)."""
        if not self.supported:
            raise self._unsupported()
        return self._model.frame_stream()


class Speech2Embeddings(Interface):
    name = "speech2embeddings"
    summary = ("audio in, embedding vectors out: one per clip (speaker embedding) or one per frame "
               "(frame-level features)")

    def _infer(self, waveform: Sequence[float]) -> "list[float] | FrameEmbeddings":
        """Embed a clip: mono floats in [-1, 1] at the model's own `contract["sample_rate"]`.

        What comes back depends on the file's declared granularity. A `clip` embedder (a speaker model)
        returns its one vector, as a plain list. A `frame` embedder (a frame-level feature extractor)
        returns a `FrameEmbeddings`: one row per encoder frame, with the time each starts at.

        The values are the model's own, unnormalised. Comparing two -- cosine similarity, for a speaker
        embedding -- or training a detector on frames is the caller's, since a threshold is a property
        of the application.
        """
        return self._model.embed(waveform)

    def stream(self) -> FrameStream:
        """Embed live (`FrameStream`), for a frame embedder: each frame's vector once its audio is in."""
        if not self.supported:
            raise self._unsupported()
        return self._model.frame_stream()


class Text2Speech(Interface):
    name = "text2speech"
    summary = "text (or phonemes) in, a waveform out"

    def _infer(self, text: str | None = None, *, phonemes: str | Sequence[int] | None = None,
               tokens: Sequence[int] | None = None, steps: int | None = None,
               seed: int = 0, language: str | None = None, voice: Any = None,
               reference: Any = None, reference_text: str | None = None,
               sample_rate: int = 16000, **driver_inputs) -> Audio:
        """Three ways in, and which ones a given model accepts is a property of the model.

        `text` needs a front end that can encode it: `"vocab"` encodes it directly, `"phonemes"` runs a
        G2P first and encodes that. A model declaring neither has no text door -- that is a step
        happening outside the engine, and the error says so rather than pretending.

        `phonemes` takes either the STRING a G2P produced (encoded here, through the model's own table
        and its own BOS/EOS assembly) or ids already encoded. `tokens` is the third and lowest: ids
        passed through untouched, assembly included, which is what the driver's own header documents.

        `voice` picks a voice for a model that takes voice files (`model.voices` lists them): a name, a
        path, or a :class:`loom.Voice`. Omitted, the model's own default is used. Its inputs go to the
        driver, and a driver input you pass explicitly still wins.

        `reference` + `reference_text` clone the voice in a clip, for a model that clones by
        IN-FILLING (F5-TTS): the clip's samples at the model's own rate (or a mono :class:`Audio` at
        that rate) and what it says, word for word. See `_reference_inputs` for the join.

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
        inputs = self._inputs(text, phonemes=phonemes, tokens=tokens, steps=steps, seed=seed,
                              language=language, voice=voice, reference=reference,
                              reference_text=reference_text, **driver_inputs)
        samples = self._model.infer(**inputs)
        if not isinstance(samples, list):
            raise TypeError(
                f"this model's driver returned {type(samples).__name__} rather than a waveform. Its "
                f"declared output kind is audio, so either the export or the driver is wrong -- "
                f"`model.driver_source` documents what `infer` actually returns."
            )
        return Audio(samples=[float(s) for s in samples], sample_rate=self._rate(sample_rate, stacklevel=4))

    def _inputs(self, text: str | None = None, *, phonemes: str | Sequence[int] | None = None,
                tokens: Sequence[int] | None = None, steps: int | None = None, seed: int = 0,
                language: str | None = None, voice: Any = None, reference: Any = None,
                reference_text: str | None = None, **driver_inputs) -> dict[str, Any]:
        """The driver inputs `_infer` documents, built and not yet run -- shared with `stream`."""
        contract = self._model.contract
        inputs: dict[str, Any] = {}
        if voice is not None:
            inputs.update({name: list(values) for name, values in self._model.voice(voice).inputs.items()})
        inputs.update(driver_inputs)
        if reference is not None or reference_text is not None:
            inputs.update(self._reference_inputs(text, phonemes, tokens, reference, reference_text,
                                                 contract, driver_inputs))
        elif self._reference_mode() == self.INFILL and not {"waveform", "text_ids"} & set(driver_inputs):
            # Said HERE rather than left to the driver, which refuses too but can only name its input.
            raise TypeError(
                f"{self._model.path.name} has no voice of its own: it clones the voice in a clip. Pass "
                f"reference=<samples at {contract.get('sample_rate') or 'its'} Hz> and "
                f"reference_text=<what the clip says, word for word> beside the text."
            )
        else:
            ids = self._resolve_ids(text, phonemes, tokens, language, contract)
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
        return inputs

    def stream(self, text: str | None = None, *, phonemes: str | Sequence[int] | None = None,
               tokens: Sequence[int] | None = None, sample_rate: int = 16000, **options) -> "SpeechStream":
        """`infer`, handed back a piece at a time: iterate the result for one :class:`Audio` per piece.

        Takes exactly `infer`'s arguments. **Two ways it can be cut, and `exact` says which:**

        * A file whose driver streams (`model.can_stream_speech`: Pocket-TTS, Soprano, VoxCPM2,
          CosyVoice3, LFM2.5-Audio, Voxtral-4B-TTS, Chatterbox, re-exported) is cut by the driver, and the pieces joined are `infer`'s waveform. `exact` is True. Soprano cuts
          at its hidden rows, each piece decoded over a window holding the decoder's whole receptive
          field as its reference's `infer_stream` does, so the join is sample for sample: 64 ms of audio
          first, then pieces growing to 1 s (`piece_tokens=` fixes the row count, 0 is a sentence). Pocket-TTS cuts at its latent frames, decoded with the decoder's state carried between
          calls -- 80 ms of audio first, then pieces growing to 640 ms (`piece_frames=` fixes the count,
          0 is a sentence chunk) -- and the join is within f32 rounding (~1e-6) of `infer`. VoxCPM2 cuts
          at its latent patches the same way: 160 ms first, then pieces growing to 1.28 s
          (`piece_patches=` fixes the count, 0 is the whole clip); unlike `infer` it never retries a
          generation that ran out of budget, as upstream's streaming mode does not. LFM2.5-Audio cuts
          at its audio frames, each piece detokenized over a window holding the detokenizer's receptive
          field: 60 ms first, then pieces growing to 640 ms (`piece_frames=` fixes the count, 0 is the
          whole turn), within f32 rounding of `infer`. Voxtral-4B-TTS cuts at its frames the same way,
          its causal codec decoding each piece from 32 frames before it: 80 ms first, then pieces growing
          to 640 ms (`piece_frames=`). CosyVoice3 cuts at
          its reference's hops of speech tokens (the first 1 to 2 s of audio, then 2 s, then 4 s),
          re-running the flow over every token so far in its STREAMING mode, as upstream's
          `tts(stream=True)` does: the join is `infer(..., streaming=1)`, not the default full-attention
          `infer`, whose audio differs -- with the same `seed`, since both draw in one order (a file
          exported before loom.cpp ADR-112's 2026-11-12 amendment does not: pin `draws`, `noise` and
          `nsf_noise` there). Even against `streaming=1` it matches on the mel (~1e-3), not sample for
          sample: the vocoder's sine phase (a running sum of F0) turns that rounding into O(1) sample
          differences. Chatterbox cuts at hops of speech tokens the same way (1 s, then 2 s, then 4 s),
          re-running its flow over every token so far with the last 3 as look-ahead, as CosyVoice2's
          stream does with the hook Chatterbox kept; its flow attends over the whole utterance, so no
          `infer` is the join: the tokens are `infer`'s (a `seed` draws them alike both ways), and the
          audio a little different -- nearer `infer`'s than `infer` under another seed's noise is.
          `piece_tokens=` fixes the hop, 0 is one piece. Each hop re-runs the flow over everything so
          far, so the whole stream costs more than `infer` (twice as much for a 3.5 s sentence on a CPU),
          while its first piece comes about when the first second of tokens is drawn.
        * Any other model is cut HERE, by splitting `text` into sentences and calling `infer` on each.
          Each piece is that sentence's own audio, and the pieces joined differ from one `infer` over
          the whole text wherever the model would have carried prosody across a sentence boundary.
          `exact` is False. Given `phonemes=` or `tokens=` there is no text to split, and the stream
          is one piece.
        """
        if not self.supported:
            raise self._unsupported()
        rate = self._rate(sample_rate, stacklevel=3)
        if self._model.can_stream_speech:
            inputs = self._inputs(text, phonemes=phonemes, tokens=tokens, **options)
            return SpeechStream(self._model._open_speech_stream(inputs), rate, exact=True)
        if text is not None and phonemes is None and tokens is None and options.get("reference") is None:
            pieces = [{"text": sentence} for sentence in split_sentences(text)]
        else:
            pieces = [{"text": text, "phonemes": phonemes, "tokens": tokens}]
        return SpeechStream(_SentencePieces(self, pieces, options), rate, exact=False)

    def _rate(self, sample_rate: int, stacklevel: int) -> int:
        declared = int(self._model.contract.get("sample_rate") or 0)
        if not declared:
            warnings.warn(
                f"{self._model.path.name} declares no sample rate, so {sample_rate} Hz is used. If that "
                f"is not this model's rate the audio will play at the wrong speed rather than fail -- "
                f"pass sample_rate= if you know it, or re-export the model declaring it.",
                RuntimeWarning, stacklevel=stacklevel,
            )
        return declared or int(sample_rate)

    #: What a model that clones by in-filling declares as `loom.tts.reference`. An hparam rather than
    #: a contract field by ADR-020's split: it does not choose the door (that is still text2speech), it
    #: says how the host builds this door's inputs.
    INFILL = "infill"

    def _reference_inputs(self, text, phonemes, tokens, reference, reference_text, contract,
                          driver_inputs) -> dict[str, Any]:
        """The driver inputs for cloning the voice in `reference`, which says `reference_text`.

        **The join is the reference's own and `loom_cli --ref-text`'s:** the model in-fills ONE
        spectrogram whose first frames are the clip, so its text is the transcript FOLLOWED by the
        text to speak, with a space between them when the transcript does not end in one (without it
        the transcript's last word runs into the text's first, and the model says them as one word).
        `n_ref_text` is how many of the ids are the transcript's -- nothing in the ids marks the join,
        and the model's duration estimate is a ratio of the two lengths.

        Which models take this is READ off the file (`loom.tts.reference`), never inferred: a model that
        does not in-fill would ignore a reference it was handed and speak in its own voice, and that
        is a wrong answer rather than an error.
        """
        if reference is None or reference_text is None:
            raise TypeError(
                "reference= and reference_text= go together: the clip, and what it says word for word. "
                "A model that clones by in-filling needs both."
            )
        if text is None or phonemes is not None or tokens is not None:
            raise TypeError(
                "a reference clones a voice for TEXT: pass text= with it, not phonemes= or tokens=. The "
                "transcript is joined to the text before either is encoded."
            )
        if not str(reference_text).strip():
            raise ValueError(
                "reference_text is empty. It is what the clip says, word for word -- the model aligns "
                "the clip's frames to it, so an empty one has nothing to align."
            )
        mode = self._reference_mode()
        if mode != self.INFILL:
            voices = contract.get("voices") or []
            other = (f" It takes voice files instead: voice= one of {voices}, or a file you made."
                     if voices else "")
            raise UnsupportedTask(
                f"{self._model.path.name} does not declare that it clones from a reference clip and "
                f"its transcript (`loom.tts.reference` is {mode or 'absent'}).{other} A file exported "
                f"before this was declared needs re-exporting with a current loom-exporter."
            )
        clash = sorted({"waveform", "text_ids", "n_ref_text"} & set(driver_inputs))
        if clash:
            raise TypeError(
                f"reference= builds {clash} itself; passing them as well would be two answers to one "
                f"question. Use one spelling or the other."
            )
        prompt = str(reference_text)
        if not prompt.endswith(" "):
            prompt += " "
        ref_ids = self._model.tokenize(prompt)
        all_ids = self._model.tokenize(prompt + text)
        return {
            "waveform": self._reference_samples(reference, contract),
            "text_ids": [float(i) for i in all_ids],
            "n_ref_text": float(len(ref_ids)),
        }

    def _phoneme_style(self) -> str:
        """`loom.tts.phoneme_style`, or "" for a model that declares none (every export before it). An
        hparam rather than a contract field by ADR-020's split, like `tts.reference`: it says how the host
        builds this door's input, not which door it is."""
        try:
            return str(self._model.hparam("tts.phoneme_style", "str"))
        except Exception:
            return ""

    def _reference_mode(self) -> str:
        try:
            return str(self._model.hparam("tts.reference", "str"))
        except Exception:
            return ""

    def _reference_samples(self, reference, contract) -> list[float]:
        """The clip as samples at the model's rate. An `Audio` carries its rate, so a mismatch is
        refused rather than played at the wrong speed into the model; plain samples are taken to be at
        the declared rate, which is what the argument documents. Resampling is the caller's: this
        package has no audio dependency, and a resampler chosen here would be a quality decision made
        silently."""
        declared = int(contract.get("sample_rate") or 0)
        if isinstance(reference, Audio):
            if reference.channels != 1:
                raise ValueError(f"the reference clip has {reference.channels} channels; pass it mono.")
            if declared and reference.sample_rate != declared:
                raise ValueError(
                    f"the reference clip is at {reference.sample_rate} Hz and this model reads "
                    f"{declared} Hz. Resample it first (librosa.load(path, sr={declared}) does both)."
                )
            reference = reference.samples
        samples = [float(s) for s in reference]
        if not samples:
            raise ValueError("the reference clip is empty.")
        return samples

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
            lang = language or (contract.get("languages") or ["en"])[0]
            spoken = phonemizers.phonemize(text, alphabet=alphabet, language=lang)
            # The conventions the checkpoint was trained in, when it declares them (`phonemizers.fold`).
            return self._model.tokenize(phonemizers.fold(spoken, self._phoneme_style(), language=lang))
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


class Text2Codes(Interface):
    name = "text2codes"
    summary = "text in, neural-codec tokens out -- an AR LM that speaks through a codec"

    def _infer(self, text: str | None = None, *, tokens: Sequence[int] | None = None,
               max_new_tokens: int | None = None, language: str | None = None, voice: Any = None,
               **driver_inputs) -> list[list[int]]:
        """Generate codec tokens for a sentence.

        **The other half of the pair is a second model**, and that is the whole shape of this door:
        what comes back is not audio, it is what a codec turns into audio. Chaining them is two calls
        and the array between them --

            codes = dia.text2codes.infer("[S1] Hello world.")
            audio = dac.codes2speech.infer(codes)

        -- which is what ADR-022 decided and why the codes are the return value rather than an
        implementation detail. A caller who wants to cache them, stream them, or decode them through a
        different codec has them; a caller who wants audio makes the second call.

        `max_new_tokens` counts **audio frames**, not decoder rows. Those differ by the model's delay
        pattern -- an AR codec LM offsets codebook *k* by *k* steps, so N frames takes N + max(delay)
        rows -- and rows are an artefact of that pattern rather than anything a caller asked for. It is
        passed on only when named, so a model's own declared ceiling applies by default.

        Two ways in, the same ladder every other door offers: `text` goes through the model's own
        vocabulary, `tokens` are ids a caller already holds.

        `voice` clones a voice for a model that takes voice files (`model.voices` lists them): a name,
        a path, or a :class:`loom.Voice` -- `Text2Speech`'s door, the same resolution. For MOSS-TTS a
        voice is one or more reference clips already encoded to codes by
        `loom_exporter.moss_tts_voices`. Its inputs go to the driver, and one you pass explicitly still
        wins.
        """
        inputs = self._inputs(text, tokens=tokens, max_new_tokens=max_new_tokens, language=language,
                              voice=voice, **driver_inputs)
        flat = self._model.infer(**inputs)
        if not isinstance(flat, list):
            raise TypeError(
                f"this model's driver returned {type(flat).__name__} rather than a run of codes. Its "
                f"declared output kind is audio_codes, so either the export or the driver is wrong -- "
                f"`model.driver_source` documents what `infer` actually returns."
            )
        return self._as_frames(flat)

    def _inputs(self, text: str | None = None, *, tokens: Sequence[int] | None = None,
                max_new_tokens: int | None = None, language: str | None = None, voice: Any = None,
                **driver_inputs) -> dict[str, Any]:
        """The driver inputs `_infer` documents, built and not yet run -- shared with `stream`."""
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

        inputs: dict[str, Any] = {}
        if voice is not None:
            inputs.update({name: list(values) for name, values in self._model.voice(voice).inputs.items()})
        inputs.update(driver_inputs)
        inputs["tokens"] = [float(i) for i in ids]
        if max_new_tokens is not None:
            inputs["max_new_tokens"] = float(max_new_tokens)
        if language is not None:
            inputs["language"] = float(self._language_index(language))
        return inputs

    def stream(self, text: str | None = None, **options) -> "CodesStream":
        """`infer`, handed back as it is drawn: iterate the result for each piece's frames (a list of
        rows, one frame per piece for Qwen3-TTS), or hand it straight to the codec --

            for audio in codec.codes2speech.stream(lm.text2codes.stream("Hello there.")):
                play(audio)

        -- which is ADR-022's two calls, each streaming (loom.cpp ADR-102). Takes `infer`'s arguments; a
        file whose driver has no stream functions raises, and runs through `infer` as before.
        """
        if not self.supported:
            raise self._unsupported()
        if not self._model.can_stream_speech:
            raise RuntimeError(
                f"{self._model.path.name}'s driver has no live path (stream_open / stream_feed / "
                f"stream_finish): it was exported before streaming. `infer` runs it as before."
            )
        return CodesStream(self._model._open_speech_stream(self._inputs(text, **options)), self)

    def _language_index(self, language: str) -> int:
        """`language` as the driver's `language` input: its 1-based position in the languages the
        FILE declares, 0 meaning none. MOSS-TTS is the first to declare them -- its prompt template
        carries a language line, pre-encoded once per language by the export. A code the file does
        not declare is refused rather than dropped: silently synthesising with no language hint is a
        worse answer than an error that lists the ones it has."""
        declared = list(self._model.contract.get("languages") or [])
        if language not in declared:
            raise ValueError(
                f"language={language!r} is not one this model declares"
                + (f"; it declares {declared}" if declared else "; it declares none")
            )
        return declared.index(language) + 1

    def _as_frames(self, flat: Sequence[float]) -> list[list[int]]:
        """The driver's flat, frame-major run, cut into per-frame rows of the declared width.

        **Rows rather than the flat run, because the width is the one thing the next model checks.**
        `Codes2Speech` takes either, but a flat list of the wrong length is the one input it cannot
        recognise as wrong -- it reinterprets as a different frame count and produces audio of the
        wrong duration with nothing raised. Handing over rows moves that check to where the codes were
        produced, which is the side that knows the answer.

        An HPARAM rather than a contract field, for the reason ADR-020 gives: it is a number the HOST
        needs to build an input with. `hparam` raises when the key is absent, and absent means an
        export too old to state it -- which is not the same as a model with no codebooks, so it is
        caught here rather than defaulted.
        """
        try:
            width = int(self._model.hparam("codec.n_codebooks", "u32"))
        except Exception:
            width = 0
        if not width:
            raise UnsupportedTask(
                "this model declares no `loom.codec.n_codebooks`, so its output cannot be cut into "
                "frames. Re-export it with a current loom-exporter, or call `model.infer(...)` and "
                "reshape with the width you know its codec to have."
            )
        if len(flat) % width:
            raise ValueError(
                f"this model's driver returned {len(flat)} codes, which is not a whole number of "
                f"frames at the {width} codebooks per frame it declares. That is the export or the "
                f"driver disagreeing with the file, not a caller error."
            )
        return [[int(c) for c in flat[i:i + width]] for i in range(0, len(flat), width)]


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
                     sample_rate=declared or int(sample_rate) or 24000,
                     channels=int(self._model.contract.get("channels") or 1))

    def stream(self, frames=None, *, frames_per_push: int | None = None, sample_rate: int = 0,
               **driver_inputs) -> "CodecStream":
        """A live decode (loom.cpp ADR-102): `push` codes as they arrive and get their audio back, then
        `finish`. Given `frames` -- any iterable of codes, such as a `text2codes.stream()` -- iterating the
        result pushes each item in turn and yields its `Audio`. The audio is what one `infer` over every
        frame pushed would give, while that is one of the codec's own chunks (300 frames for Qwen3-TTS's
        12 Hz codec); past it, `infer` is the reference's chunked decode and the stream is not.

        Iterating gathers `frames` into pushes that GROW -- 1 frame, then 2, 4 and 8 from then on -- so
        the first audio waits for one frame and the rest pays little for the cut: on the dev box one
        frame per push decoded at RTF 2.56 and four or more beat the one-pass decode (1.82 against
        1.90). `frames_per_push` fixes the count.
        """
        if not self.supported:
            raise self._unsupported()
        if not self._model.can_stream_speech:
            raise RuntimeError(
                f"{self._model.path.name}'s driver has no live path (stream_open / stream_feed / "
                f"stream_finish): it was exported before streaming. `infer` decodes it as before."
            )
        declared = int(self._model.contract.get("sample_rate") or 0)
        return CodecStream(self._model._open_codec_stream(driver_inputs), self,
                           declared or int(sample_rate) or 24000,
                           int(self._model.contract.get("channels") or 1), frames, frames_per_push)

    def _absent_code(self):
        """The id this codec decodes as "codebook not in the row", or None when it declares none.

        A residual quantizer's decode of its first k codebooks is a PREFIX of the full sum, so a codec
        declaring this can take rows narrower than its width -- the LM that feeds it emits fewer
        codebooks than it has (MOSS-TTS: 12 of MOSS-Audio-Tokenizer's 32). Absent for every codec
        whose rows must be full."""
        try:
            return int(self._model.hparam("codec.absent_code", "u32"))
        except Exception:
            return None

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
            absent = self._absent_code()
            if absent is not None and width:
                # Narrower rows are a prefix; the rest of each is filled with the absent id. A row
                # WIDER than the codec is still an error below, as is any row on a codec without one.
                rows = [r + [absent] * (width - len(r)) if len(r) < width else r for r in rows]
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
Image2Class = _planned("image2class", "image classification")
Text2Embeddings = _planned("text2embeddings", "text embedding")
Image2Embeddings = _planned("image2embeddings", "image embedding")
Image2Boundingbox = _planned("image2boundingbox", "object detection")
Image2Segmentationmask = _planned("image2segmentationmask", "segmentation")


#: Every interface a `Model` carries, in the order `capabilities` and `repr` report them. The
#: implemented ones first, because that is the order a reader cares about.
ALL_INTERFACES = (
    Text2Text, Speech2Text, Text2Speech, Text2Class, Text2Codes, Codes2Speech,
    Speech2Class, Speech2Embeddings,
    Speech2Speech, Text2Image, Image2Text, Speech2Image, Image2Speech,
    Image2Class,
    Text2Embeddings, Image2Embeddings,
    Image2Boundingbox, Image2Segmentationmask,
)
