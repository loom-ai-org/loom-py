"""The published model cards, executed against the artefacts they describe.

**What this is for.** Every other test in this repo asks whether the code is right. This one asks
whether the DOCUMENTATION is -- specifically the `README.md` that ships beside each GGUF and becomes
the model card on the Hub, which is the first and often only thing a user runs. A card is the one
piece of a release that is written by hand, published verbatim, and never executed. It drifts
silently: an API gains an argument, a door moves, a model is re-exported without the capability its
card advertises, and nothing fails until somebody copies the snippet.

That is not hypothetical here. `qwen3-asr-0.6b` and `granite-speech-4.0-1b` shipped cards
advertising `speech2text.infer(...)` while any clip that was not a whole number of encoder chunks
died inside the driver with a `RESHAPE` error (fixed in loom.cpp#18) -- a defect no CI test could
see, because the CI suite never loads a real model and the gate suite drove the subgraphs directly.

**So the card is the specification, and this runs it.** The `python` blocks are extracted and
executed in order, in one namespace, exactly as a reader would follow them top to bottom. One kind of
substitution is made:

    loom.Model.from_pretrained("loom-ai-org/<repo>")  ->  loom.Model.from_file("<local .gguf>")

because a release gate must test the artefact about to be published, not the one already on the Hub.
It applies to EVERY repo a block names, resolved against the staging tree -- one card legitimately
loads two models, since an AR codec-token LM needs a codec to become audible. Everything else runs as
printed. A card that needs an argument it does not show, or shows one the API no longer takes, fails
here.

**Three questions, per the family:**

* *does it work* -- the block runs to completion;
* *is it consistent* -- a model whose export declares greedy decoding gives the same answer twice;
* *is it right* -- and for a TTS model this is the only question that matters, because correlation is
  not the test. Kokoro once matched PyTorch at cosine 0.996 and shipped unintelligible
  (loom.cpp Retro-006), so synthesised audio is TRANSCRIBED BACK by a standard ASR model and
  compared with the words the card asked for. All five TTS cards say "hello world", which also makes
  them comparable with each other. A token classifier gets the same treatment one modality over: its
  card labels a fixed sentence and the entities are reconstructed from the labels and checked, because
  "the block ran" is equally satisfied by a model that answers `O` to everything. A codec decoder is
  graded on its output LENGTH, which is what silently broke on the first one -- and a codec-token LM
  is graded by chaining it through that codec and transcribing the result, which is the TTS question
  asked of a model that emits no audio of its own.

**Running it:**

    export LOOM_MODEL_CARDS=~/Dev/loom/hf-models      # the staging tree, cards beside GGUFs
    export LOOM_CARD_VOICES=<dir of <card>.gguf>      # optional: jfk.wav as each cloning card's voice file
    pytest tests/gate/test_model_cards.py -q

**Which rows apply is decided by `loom.contract_of`, not by opening the model.** Every row here is
parametrised over every model and skips the ones it is not for, so a suite of N models asks that
question N times per row -- and answering it by loading the model costs the model. On the 6.4 GB
family-10 card that was half the row's memory and got a run OOM-killed (exit 137, peak 28.9 GB
against 1.9 GB free); `contract_of` reads the GGUF's KV table and stops. The two rows that go on to
need a real model now load it *after* the check rather than before, so a model this row is not for is
never opened at all.

**The voice-file row synthesises once per staged voice set**, on top of the TTS row's own synthesis, so
a model with `voices/` is loaded twice. For Voxtral-4B-TTS that is a second 16 GB load, which the 2-core
dev box cannot afford beside anything else: run that model's rows on the workstation, or deselect them
here with `-k "not voxtral"` (the accounting check then counts its rows as deselected, not skipped).

It skips cleanly without that variable, like every gate test. `pip install "loom-py-rt[phonemes]"`
additionally covers the text-in door; without it the cards' G2P lines are reported as skipped
preconditions rather than failures, because a missing optional extra is not a broken card.
"""
import gc
import math
import os
import re
import wave
from pathlib import Path

import pytest

import loom

# The reference recording, and it is IN THE REPO rather than downloaded: a gate that fetches its own
# ground truth can fail for reasons that are not about the release. 11.00 s, mono, 16 kHz.
JFK_WAV = Path(__file__).resolve().parents[2] / "vendor" / "loom.cpp" / "samples" / "jfk.wav"
JFK_WORDS = (
    "and so my fellow americans ask not what your country can do for you "
    "ask what you can do for your country"
)

# What every TTS card asks for, which is what makes the ASR oracle's expectation a constant.
TTS_WORDS = "hello world"

# The sentence every token-classification card labels, and the entities that have to come back out of
# it. Same device as TTS_WORDS one task over: fixing the input in the CARD is what lets the expectation
# here be a constant rather than a second model run grading the first.
#
# THE SPANS, NOT THE LABEL SEQUENCE, and that is the point of writing it this way. A per-token
# expectation would have to know how this checkpoint's vocabulary splits "Wolfgang" -- which differs
# between a cased and an uncased card and is not what anyone wants to assert. Reconstructing B-/I-
# runs into spans asks the question a user actually has: did it find the right entities.
CLASSIFY_ENTITIES = {("wolfgang", "PER"), ("berlin", "LOC")}

# The same device for the other reading of family 12's door. A punctuation checkpoint's classes are
# MARKS rather than span tags, so there are no spans to reconstruct and the question is simply whether
# the right marks landed on the right words: `berlin.` ends a sentence and `it?` ends a question. Its
# card labels the SAME sentence the NER card does, which is what makes these two constants comparable.
#
# THE PIECE, NOT THE WORD, and deliberately: a SentencePiece vocabulary splits `wolfgang` into three
# and the mark belongs to a word's LAST piece, so a word-final piece is exactly what a mark attaches
# to. Reconstructing words here would be re-implementing the loop the card publishes.
CLASSIFY_MARKS = {("berlin", "."), ("it", "?")}

# Family 13's expectations, against the same recording every ASR row uses (ADR-062). jfk.wav is 11 s of
# one man speaking English with pauses, so each of the three classifier kinds has an answer that is a
# CONSTANT rather than a second model's opinion: the language is English, most frames hold speech, and
# a frame model's rows span the clip at the rate the file declares. Measured 2026-10-01 on the rc13
# exports: MarbleNet 73% speech frames (zero on silence), pyannote 70% speaker frames, ECAPA
# `en: English` at 0.82.
#
# A frame model's row count is `seconds * frame_rate` less what its receptive field consumes at the
# edges, which is a few frames and does not grow with the clip (pyannote: 589 for 592.6, 115 for 118.5).
FRAME_ROW_SLACK = 5
# The fraction of jfk.wav's frames a VAD or a segmentation model must place in a speech class. Both sit
# near 0.7; a broken export answers all non-speech (a dead head) or a constant row, and lands far away.
MIN_SPEECH_FRACTION = 0.5
# A speaker embedding is graded by COMPARISON, which is what the vector is for: the two halves of one
# recording must score as one speaker, and the same recording resampled 1.4x (higher and faster -- a
# different voice to the model) must not. Measured: 0.69 and 0.06. A constant output -- the classic
# broken embedder -- scores 1.0 on both and fails the second.
MIN_SAME_SPEAKER = 0.5
# Every precision a multi-file repo carries, against its largest file: mean per-frame cosine. Q4_1
# wakehubert-tiny measured 0.987 against PyTorch, Q4_0 0.981 (2026-10-03); a broken file lands far below.
MIN_PRECISION_AGREEMENT = 0.95
MAX_OTHER_SPEAKER = 0.3


# The model that reads TTS output back. Whisper rather than a NeMo model because it is the one every
# card set already depends on for the ASR examples, and because its own card is checked here too --
# a broken oracle would fail its own row first, which is the ordering you want.
ORACLE = "whisper-small"

# PER-MODEL BASELINES, MEASURED, because one global ceiling cannot serve this family. Six of the
# seven transcribe jfk.wav PERFECTLY; GigaAM scores 0.50 and is not broken. A ceiling loose enough
# for GigaAM (0.6) would let Whisper rot from 0.00 to 0.30 unnoticed, and a tight one fails a model
# that works. What this gate is for is BREAKAGE, not quality -- a broken model scores about 1.0
# (silence, or noise) -- and breakage is per-model distance from where that model actually sits.
#
# Measured 2026-08-31 against samples/jfk.wav, on the rc7 exports:
ASR_BASELINE = {
    "conformer-ctc-small":    0.00,
    # "american" for "americans" and "as" for "ask": 2 in 22. NeMo's own forward decodes the identical
    # string from the identical audio (2026-10-01), so this is the checkpoint, not the port.
    "citrinet-1024":          0.09,
    # Word-perfect, with punctuation and casing; its card asks for the default English target.
    "canary-1b-v2":           0.00,
    # Word-perfect, punctuation and casing included, at both sizes -- and identical in ids to
    # transformers' own generate on jfk.wav and on all 73 LibriSpeech-dummy utterances (2026-10-02).
    "moonshine-streaming-tiny":  0.00,
    "moonshine-streaming-small": 0.00,
    # Word-perfect, punctuation and casing included; each matches its own reference id for id on jfk.wav
    # (Kyutai's moshi, at 24 kHz through the card's resample; liquid-audio's generate_sequential), 2026-10-02.
    "kyutai-stt-1b-en-fr":       0.00,
    "lfm2.5-audio-1.5b-asr":     0.00,
    "granite-speech-4.0-1b":  0.00,
    "parakeet-rnnt-0.6b":     0.00,
    # Word-perfect, punctuation and casing included, with the language left to `auto`; identical in ids
    # to transformers' generate (sdpa) on jfk.wav and on a 60 s clip (2026-10-10). Its `<en-US>` tags
    # are control ids, so they do not reach `text` to be counted as words.
    "nemotron-3.5-asr-streaming-0.6b": 0.00,
    "parakeet-tdt-0.6b":      0.00,
    "whisper-small":          0.00,
    # Russian-first (it declares `ru, en`), so English costs it. The transcript is unmistakably the
    # right utterance -- "my fellow americans ... your country can do for you" -- spelled through a
    # recogniser trained elsewhere. Not a defect, and not something to tighten.
    "gigaam-v3-rnnt":         0.50,
    # WAS 0.18, AND THAT 0.18 WAS ENTIRELY THE CONTROL MARKERS -- `language English<asr_text>` counted
    # as four spurious words against a 21-word reference, while the speech itself was already perfect.
    # They no longer leak (re-measured 2026-09-01: the card returns the utterance and nothing else),
    # so the to-do is closed and the baseline follows it down. Tightening it is the POINT, not
    # bookkeeping: left at 0.18 the ceiling stays 0.33, the markers cost about 0.18 to reinstate, and
    # the one regression this row exists to catch could come back and still pass.
    "qwen3-asr-0.6b":         0.00,
    # Family 4, and both rows are the checkpoint rather than the export. These are character-level CTC
    # models with no language model behind them, so what they produce is a phonetic spelling of what was
    # said, which is right and is not what a decoder would produce. HuBERT-Large gets the utterance
    # exactly. data2vec-Base inserts one word ("and so A my fellow"), 1 error in 21 -- `transformers`
    # produces the identical insertion from the identical audio, so tightening this would be measuring
    # the reference rather than the port.
    "hubert-large-ls960-ft":      0.00,
    "data2vec-audio-base-960h":   0.05,
    # Family 5, and the SenseVoice row is this table's own lesson applied a second time. It first
    # measured 0.23 -- word-perfect speech behind four spurious "words", because SenseVoice emits its
    # detected language, emotion, audio event and text-normalization mode as ordinary vocabulary pieces
    # before the transcript. Recording 0.23 would have set a ceiling of 0.38 on a model that actually
    # sits at 0.00, which is the failure the qwen3-asr comment above describes. The export declares
    # those 171 tag ids as `loom.asr.control_ids` instead, so `transcribe` strips them and
    # `detokenize` still returns them; the baseline follows the fix down rather than absorbing it.
    "sensevoice-small":           0.00,
    "paraformer-zh":              0.00,
}

# How far past its own baseline a model may drift. Wide enough to absorb the punctuation and casing
# an ASR model is entitled to vary ("americans" / "American's" are the same claim), narrow enough
# that a model which stops recognising speech at all cannot hide inside it.
ASR_MARGIN = 0.15

# The TTS side keeps a single ceiling, because there the reference is one word pair and the failure
# it guards against is total: Kokoro shipped at cosine 0.996 against PyTorch and transcribed to
# nothing recognisable (loom.cpp Retro-006). A 1 s "hello world" is a hard clip for any recogniser,
# so this is deliberately generous -- it separates "said the words" from "said nothing".
MAX_WER_TTS = 0.50

# Audio that is not silence and not clipped. A TTS model that emits zeros transcribes to "" and would
# otherwise be caught only by the WER; this says which of the two went wrong.
MIN_PEAK, MAX_PEAK = 0.01, 1.001


def _cards_dir():
    root = os.environ.get("LOOM_MODEL_CARDS")
    if not root:
        pytest.skip("LOOM_MODEL_CARDS is not set; it names the tree holding <model>/README.md + .gguf")
    path = Path(root).expanduser()
    if not path.is_dir():
        pytest.skip(f"{path} is not a directory")
    return path


def _discover():
    """Every directory carrying both a card and a GGUF, as (name, gguf, readme)."""
    root = os.environ.get("LOOM_MODEL_CARDS")
    if not root or not Path(root).expanduser().is_dir():
        return []
    out = []
    for d in sorted(Path(root).expanduser().iterdir()):
        readme, ggufs = d / "README.md", sorted(d.glob("*.gguf"))
        if d.is_dir() and readme.is_file() and ggufs:
            out.append((d.name, max(ggufs, key=lambda p: p.stat().st_size), readme))
    return out


DISCOVERED = _discover()
NAMES = [n for n, _, _ in DISCOVERED] or ["<no LOOM_MODEL_CARDS>"]


def _entry(name):
    for n, gguf, readme in DISCOVERED:
        if n == name:
            return gguf, readme
    pytest.skip(f"{name} not present in LOOM_MODEL_CARDS")


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate, lowercased and stripped of punctuation.

    Levenshtein over words rather than characters: the question is whether the model said the right
    WORDS, and a character metric would score "balloon" against "loom" as nearly right.
    """
    norm = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
    r, h = norm(reference), norm(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def python_blocks(readme: Path):
    return re.findall(r"```python\n(.*?)```", readme.read_text(), re.S)


def localise(block: str, gguf: Path) -> str:
    """The one substitution: publish-time `from_pretrained` becomes the local artefact.

    A release gate has to run what is about to be published. Left alone, every card would download
    the PREVIOUS release from the Hub and pass while the new GGUF beside it was broken -- which is
    the exact failure mode this whole file exists to prevent, so it would be a poor thing to inherit.

    **Every repo the block names is resolved, not just this card's own**, and one card needs that:
    an AR codec-token LM emits tokens and a codec turns them into audio, so `dia-1.6b`'s snippet
    loads `dac-44khz-loom` as well. Rewriting both to this card's own GGUF -- which is what a single
    blanket substitution did -- would hand the codec a text model and fail in a way that looks like a
    broken card. A repo the staging tree does not carry is left as `from_pretrained`, so it downloads
    and the card still runs; that is the honest fallback, since a release cannot be blocked on a
    model it is not publishing.

    **A file name, when the block passes one, is resolved too**: a repo carrying several precisions
    of one model (wakehubert-tiny) is loaded as `from_pretrained(repo, "<file>.gguf")`, and that file
    is the staged one beside the card -- not the largest, which is what `gguf` names.
    """
    def replace(match: "re.Match") -> str:
        slug = match.group(1).split("/")[-1].removesuffix("-loom")
        filename = match.group(2)
        if filename:
            staged = gguf.parent.parent / slug / filename
            return f"loom.Model.from_file({str(staged)!r})" if staged.is_file() else match.group(0)
        if slug == gguf.stem:
            return f"loom.Model.from_file({str(gguf)!r})"
        sibling = gguf.parent.parent / slug / f"{slug}.gguf"
        if sibling.is_file():
            return f"loom.Model.from_file({str(sibling)!r})"
        return match.group(0)

    block = re.sub(
        r"loom\.Model\.from_pretrained\(\s*['\"]([^'\"]+)['\"]\s*(?:,\s*['\"]([^'\"]+\.gguf)['\"]\s*)?\)",
        replace,
        block,
    )

    # A data file the card loads from its OWN repo (sanoTTS's lexicons: `hf://loom-ai-org/<repo>/<path>`)
    # is the staged copy too, for the same reason: the Hub holds the previous release, or nothing yet.
    def replace_hf(match: "re.Match") -> str:
        slug = match.group(1).removesuffix("-loom")
        staged = gguf.parent.parent / slug / match.group(2)
        return repr(str(staged)) if staged.is_file() else match.group(0)

    return re.sub(r"['\"]hf://loom-ai-org/([^/'\"]+)/([^'\"]+)['\"]", replace_hf, block)


def produced(ns, *attrs):
    """The last value the card bound that has all of `attrs`, or None.

    BY SHAPE, NOT BY NAME. Every card happens to call it `result` or `audio` today, but that is a
    house style, not a contract -- and a gate that greps for a variable name would start silently
    testing nothing the day a card renamed one. Namespaces preserve insertion order, so scanning in
    reverse takes the LAST one bound, which is what a reader following the card top to bottom ends
    up holding.
    """
    for value in reversed(list(ns.values())):
        if all(hasattr(value, a) for a in attrs):
            return value
    return None


@pytest.fixture(scope="session")
def jfk():
    """The reference utterance as a mono float list at 16 kHz."""
    if not JFK_WAV.is_file():
        pytest.skip(f"{JFK_WAV} is missing (the engine submodule is not checked out)")
    with wave.open(str(JFK_WAV)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000
        raw = w.readframes(w.getnframes())
    return [int.from_bytes(raw[i:i + 2], "little", signed=True) / 32768.0
            for i in range(0, len(raw), 2)]


@pytest.fixture(scope="session")
def oracle():
    """The ASR model that reads TTS output back."""
    for name, gguf, _ in DISCOVERED:
        if name == ORACLE:
            return loom.Model.from_file(str(gguf))
    pytest.skip(f"{ORACLE} is not in LOOM_MODEL_CARDS; it is the oracle for every TTS row")


# Every namespace `run_card` built, emptied when its test ends. A test that SKIPS after running a card
# -- a voice-cloning card stops at the reader's own voice file, which is a skip -- leaves its exception
# on the report, the traceback keeps the test's frame, and the frame keeps the namespace: MOSS-TTS's
# 16.8 GB model and its codec stayed resident into the next row, which loaded them again and was
# OOM-killed at 27.3 GB. Emptying the dict frees the models whatever still holds the frame.
_CARD_NAMESPACES = []


@pytest.fixture(autouse=True)
def _release_card_namespaces():
    yield
    for ns in _CARD_NAMESPACES:
        ns.clear()
    _CARD_NAMESPACES.clear()
    gc.collect()


def _card_voice(name):
    """`$LOOM_CARD_VOICES/<name>.gguf` -- the gate's voice file for one card -- or None."""
    root = os.environ.get("LOOM_CARD_VOICES")
    if not root:
        return None
    path = Path(root).expanduser() / f"{name}.gguf"
    return path if path.is_file() else None


def run_card(name, gguf, readme, jfk, tmp_path, monkeypatch):
    """Execute every block of one card, in order, in one namespace; return that namespace.

    One namespace and in order because that is how a card is READ -- a later block may legitimately
    use a name an earlier one bound. `audio` is seeded because a card cannot ship a recording and
    says so; it is the caller's own data, and the only name this harness invents.

    Returning the namespace is what lets the ASR oracle grade a TTS model on the audio ITS OWN CARD
    produced, rather than on a call this file reinvents -- which would be a second, unpublished
    spelling of the thing under test, and would have to guess a sample rate the card knows.
    """
    blocks = python_blocks(readme)
    assert blocks, f"{name}'s card publishes no python block, so it documents nothing runnable"
    monkeypatch.chdir(tmp_path)   # cards write out.wav; let them, somewhere disposable
    # THE READER'S OWN RECORDING, for a card that clones a voice. Such a card cannot ship one, so it
    # names a file -- `reference.wav` -- and without it the card stopped at that precondition, which
    # meant no cloning card was ever executed here. jfk.wav stands in for it on every such card (the
    # user's call, 2026-09-26: one common fixture, public domain, already this file's ASR reference),
    # and a card that uses it prints the clip's own transcript, which is what makes it a fair prompt.
    (tmp_path / "reference.wav").write_bytes(JFK_WAV.read_bytes())
    # THE READER'S OWN VOICE FILE, for a card whose reader file is one rather than a clip (MOSS-TTS,
    # CosyVoice3: `voices/me.gguf`). The same stand-in, one step further along: jfk.wav made into a voice
    # file for THAT model, since a voice file is stamped with the weights it was made for and making one
    # needs PyTorch and the upstream checkpoint -- neither of which this gate may assume.
    # `LOOM_CARD_VOICES` names a directory of `<card>.gguf`; without it such a card stops at the file,
    # as a precondition the skip message names.
    voice = _card_voice(name)
    if voice is not None:
        (tmp_path / "voices").mkdir(exist_ok=True)
        (tmp_path / "voices" / "me.gguf").write_bytes(voice.read_bytes())
    ns = {"loom": loom, "audio": jfk}
    _CARD_NAMESPACES.append(ns)
    # A PRECONDITION STOPS THE BLOCK BUT DOES NOT DISCARD WHAT IT ALREADY DID, and the first version
    # of this got that wrong in a way that silently cost real coverage. Four of the five TTS cards
    # synthesise from phonemes FIRST and only then call `set_lexicon` to demonstrate the text door.
    # Skipping out of here on the lexicon threw away the waveform the card had already produced, so
    # the ASR oracle -- the one check that matters for a TTS family -- ran on exactly one model out
    # of five and the suite still looked green. Return what was bound and let the caller judge.
    unmet = None
    for i, block in enumerate(blocks):
        try:
            exec(compile(localise(block, gguf), f"{name}#block{i}", "exec"), ns)
        except LookupError as e:
            if "orthography2ipa" not in str(e):
                raise
            unmet = f"{name} block {i} needs the [phonemes] extra: {e}"
            break
        except FileNotFoundError as e:
            # A card may legitimately tell the reader to bring a file (a lexicon it links to). That
            # is a precondition, not a broken card -- but name it, so the list stays visible.
            unmet = f"{name} block {i} needs a file the reader supplies: {e}"
            if "voices" in str(e) and _card_voice(name) is None:
                unmet += (f" -- set LOOM_CARD_VOICES to a directory holding {name}.gguf, a voice "
                          f"file made from jfk.wav, to run it")
            break
    return ns, unmet


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_the_card_runs(name, jfk, tmp_path, monkeypatch):
    """Every `python` block in the card, in order, in one namespace.

    One namespace and in order because that is how a card is READ -- a later block may legitimately
    use a name an earlier one bound. `audio` is provided because a card cannot ship a recording and
    says so; it is the caller's own data, and the only name this harness invents.
    """
    """Every `python` block in the card runs, exactly as published."""
    _cards_dir()
    gguf, readme = _entry(name)
    _, unmet = run_card(name, gguf, readme, jfk, tmp_path, monkeypatch)
    if unmet:
        pytest.skip(unmet)


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_asr_transcribes_the_reference(name, jfk, tmp_path, monkeypatch):
    """An ASR model gets the words right, against a recording checked into the repo."""
    _cards_dir()
    gguf, readme = _entry(name)
    if loom.contract_of(gguf).get("interface") != "speech2text":
        pytest.skip(f"{name} is not speech2text")

    # THE CARD'S OWN CALL, not one this file invents -- and the first version of this DID invent one,
    # passing `language="en"` to every ASR model. Six of the seven cards correctly omit it (only
    # Whisper is windowed, so only Whisper has a prompt a language token can go in), the engine
    # warned on all six, and the warning was briefly misread as the CARDS being wrong. They were not.
    # Grading what the card published makes that class of mistake unavailable: there is no second
    # spelling to get wrong, and `audio` is the reference recording seeded into the namespace.
    ns, unmet = run_card(name, gguf, readme, jfk, tmp_path, monkeypatch)
    result = produced(ns, "text", "segments")
    if result is None:
        pytest.skip(f"{name}'s card bound no transcription{' -- ' + unmet if unmet else ''}")

    assert result.text.strip(), f"{name} transcribed 11 s of speech to nothing"
    if name not in ASR_BASELINE:
        pytest.skip(f"no WER baseline recorded for {name!r}; measure it against jfk.wav and add one")
    rate = wer(JFK_WORDS, result.text)
    ceiling = ASR_BASELINE[name] + ASR_MARGIN
    assert rate <= ceiling, (
        f"{name} WER {rate:.2f} > {ceiling:.2f} (baseline {ASR_BASELINE[name]:.2f} + "
        f"{ASR_MARGIN:.2f})\n  heard: {result.text!r}"
    )


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_tts_output_is_intelligible(name, oracle, jfk, tmp_path, monkeypatch):
    """Synthesise what the card says, then READ IT BACK with an ASR model.

    Correlation against a reference implementation is not the test and never was: Kokoro matched
    PyTorch at cosine 0.996 and shipped noise (loom.cpp Retro-006). The only check that would have
    caught it is this one -- does a recogniser hear the words.
    """
    _cards_dir()
    gguf, readme = _entry(name)
    if loom.contract_of(gguf).get("interface") != "text2speech":
        pytest.skip(f"{name} is not text2speech")

    ns, unmet = run_card(name, gguf, readme, jfk, tmp_path, monkeypatch)
    # By shape, and the shape matters: `audio` is SEEDED with the reference recording, which is a
    # plain list. Only a synthesised waveform carries `sample_rate`, so a TTS card that never
    # produced one cannot slip through with this test grading the oracle on jfk.wav and passing.
    audio = produced(ns, "samples", "sample_rate")
    if audio is None:
        pytest.skip(f"{name}'s card synthesised nothing{' -- ' + unmet if unmet else ''}")
    samples = _mono(audio)
    rate = audio.sample_rate
    assert samples, f"{name} synthesised nothing"
    peak = max(abs(s) for s in samples)
    assert MIN_PEAK <= peak <= MAX_PEAK, (
        f"{name} peak {peak:.4f} outside [{MIN_PEAK}, {MAX_PEAK}] -- silence or clipping, "
        f"which is a different failure from the wrong words"
    )

    # `language="en"` is correct HERE and nowhere else in this file: the oracle is Whisper, the one
    # ASR export that is windowed and therefore has a prompt a language token can go in.
    heard = oracle.speech2text.infer(_resample_16k(samples, rate), language="en").text
    assert heard.strip(), f"{name} synthesised audio the oracle heard as nothing (peak {peak:.4f})"
    rate_wer = wer(TTS_WORDS, heard)
    assert rate_wer <= MAX_WER_TTS, (
        f"{name} said {TTS_WORDS!r}, oracle heard {heard!r} (WER {rate_wer:.2f})"
    )


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_every_staged_voice_file_fits_and_speaks(name, oracle):
    """Every `voices/*.gguf` beside the model loads into it, and one of them speaks the words.

    Nothing else here opens a staged voice file: a card's snippet names one voice at most, and most name
    none. But a voice file is stamped with what its model declares (`loom.voice.compat` and the
    architecture, loom.cpp ADR-045), so a re-export that changes either leaves every file under
    `voices/` unloadable while the card, which uses the built-in voice, still passes. Loading is cheap
    (the engine compares two strings and reads a small tensor), so ALL of them are loaded; one that is
    not the built-in voice is then synthesised and read back, which is what says the file sets the
    input the driver actually uses.
    """
    _cards_dir()
    gguf, _ = _entry(name)
    voices = sorted((gguf.parent / "voices").glob("*.gguf"))
    if not voices:
        pytest.skip(f"{name} stages no voices/*.gguf")
    if loom.contract_of(gguf).get("interface") != "text2speech":
        pytest.skip(f"{name}'s voice files are for a {loom.contract_of(gguf).get('interface')} door; "
                    f"the cloning-card rows cover those")
    model = loom.Model.from_file(gguf)
    try:
        for path in voices:
            voice = model.voice(path)
            assert voice.inputs, f"{name}: {path.name} sets no driver input"
        built_in = set(model.contract.get("voices") or [])
        pick = next((p for p in voices if p.stem not in built_in), voices[0])
        audio = model.text2speech.infer(TTS_WORDS, voice=pick.stem)
        samples = _mono(audio)
        peak = max(abs(s) for s in samples) if samples else 0.0
        assert MIN_PEAK <= peak <= MAX_PEAK, f"{name} voice {pick.stem!r}: peak {peak:.4f}"
        heard = oracle.speech2text.infer(_resample_16k(samples, audio.sample_rate), language="en").text
        rate_wer = wer(TTS_WORDS, heard)
        assert rate_wer <= MAX_WER_TTS, (
            f"{name} voice {pick.stem!r} said {TTS_WORDS!r}, oracle heard {heard!r} (WER {rate_wer:.2f})"
        )
    finally:
        del model
        gc.collect()


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_a_codec_lm_says_the_words(name, oracle, jfk, tmp_path, monkeypatch):
    """The *is it right* question for family 10, and it is the same question as for TTS one door over.

    An AR codec-token LM is graded on what its codes SOUND like once a codec has decoded them, for
    exactly the reason `test_tts_output_is_intelligible` exists: this model's codes match
    `transformers` byte-for-byte under a greedy decode, and that says nothing about the sampled one it
    actually ships with (loom.cpp Retro-006, and Retro-032 for this family's own version of it).

    **The sentence and the seed come from the card**, not from here. This checkpoint samples at
    `temperature 1.8` with classifier-free guidance and is high-variance -- some seeds give laughter
    or near-silence -- so the card names one that works, and grading the card's own output is what
    makes that a published promise rather than a private measurement. If this row fails, the card is
    telling readers to run something that does not say the words.

    It needs the codec beside it in the staging tree, which `localise` resolves; without it the card's
    second `from_pretrained` reaches the Hub and this still runs.
    """
    heard, readme = _hear_codec_lm_card(name, oracle, jfk, tmp_path, monkeypatch, music=False)
    # The expectation is the card's own sentence, read back out of it rather than restated here --
    # this family's cards do not share one line the way the TTS cards share "hello world", and a
    # constant copied into this file would be a second, unpublished spelling of what is under test.
    said = card_sentence(readme)
    assert said, f"{name}'s card passes no sentence to text2codes, so nothing can be expected of it"
    codes_wer = wer(said, heard)
    assert codes_wer <= MAX_WER_TTS, (
        f"{name} was asked for {said!r}, oracle heard {heard!r} (WER {codes_wer:.2f})"
    )


# What Whisper writes for audio it recognises as music rather than speech: a sound tag such as
# "(upbeat music)" or "[Music]", or the "♪" it puts around sung lines.
MUSIC_TAG = re.compile(r"[(\[][^)\]]*music[^)\]]*[)\]]|♪", re.IGNORECASE)


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_a_music_lm_makes_music(name, oracle, jfk, tmp_path, monkeypatch):
    """The codec-LM row for a card that makes MUSIC (`pipeline_tag: text-to-audio`, MusicGen).

    A music prompt describes a genre and has no words to read back, so the row above cannot grade it:
    MusicGen's card, working correctly, was heard as "(upbeat music)" and failed it at WER 1.0. What can
    be asked is whether the recogniser files the output under music. Measured on whisper-small
    (2026-10-10): two MusicGen takes -> "(upbeat music)" both; JFK's speech -> his words; white noise ->
    "(water splashing)"; random EnCodec codes -> "(roaring)" with a peak of 1.59, which the peak bound
    rejects as well. Speech, noise and a broken LM all fail this.
    """
    heard, _ = _hear_codec_lm_card(name, oracle, jfk, tmp_path, monkeypatch, music=True)
    assert MUSIC_TAG.search(heard), (
        f"{name} is a music card, and the oracle heard its output as {heard!r}, not as music"
    )


def _hear_codec_lm_card(name, oracle, jfk, tmp_path, monkeypatch, *, music):
    """Run a text2codes card and return what the oracle heard, plus the README. Skips a card from the
    other codec-LM row (speech vs music, by the card's `pipeline_tag`) BEFORE running it."""
    _cards_dir()
    gguf, readme = _entry(name)
    if loom.contract_of(gguf).get("interface") != "text2codes":
        pytest.skip(f"{name} is not text2codes")
    if card_makes_music(readme) != music:
        pytest.skip(f"{name} {'does not make' if music else 'makes'} music: graded by the other row")

    ns, unmet = run_card(name, gguf, readme, jfk, tmp_path, monkeypatch)
    audio = produced(ns, "samples", "sample_rate")
    if audio is None:
        pytest.skip(f"{name}'s card produced no audio{' -- ' + unmet if unmet else ''}")
    samples = _mono(audio)
    peak = max(abs(s) for s in samples)
    assert MIN_PEAK <= peak <= MAX_PEAK, (
        f"{name} peak {peak:.4f} outside [{MIN_PEAK}, {MAX_PEAK}] -- silence or clipping, which is a "
        f"different failure from the wrong words, and the one a bad seed produces"
    )

    heard = oracle.speech2text.infer(_resample_16k(samples, audio.sample_rate), language="en").text
    assert heard.strip(), f"{name} produced audio the oracle heard as nothing (peak {peak:.4f})"
    return heard, readme


def card_makes_music(readme: Path) -> bool:
    """Whether the card's Hub `pipeline_tag` is text-to-audio (music), not text-to-speech. Read from the
    card's front matter, the same published declaration the Hub files it under."""
    match = re.search(r"^pipeline_tag:\s*(\S+)\s*$", readme.read_text(), re.MULTILINE)
    return bool(match) and match.group(1) == "text-to-audio"


def card_sentence(readme: Path) -> str:
    """The sentence the card hands to `text2codes`, with this family's speaker tags stripped.

    `[S1]`/`[S2]` are real input tokens for a dialogue model -- they are what makes it one -- but no
    recogniser transcribes them, so they are not part of what the oracle should hear. Read out of the
    card rather than declared here for the reason the assertion above gives.
    """
    match = re.search(r"text2codes\.infer\(\s*[\"']([^\"']+)[\"']", readme.read_text())
    if not match:
        return ""
    return re.sub(r"\[S\d\]", " ", match.group(1)).strip()


def labels_are_spans(labels) -> bool:
    """Whether this checkpoint's classes are IOB2 SPAN tags rather than per-token marks.

    Read off the label set THE FILE DECLARES, which is the only authority for it -- both kinds of model
    reach `text2class` through one door and return the identical shape, and the difference is entirely
    what a label MEANS. A gate keyed on the model's name instead would have to be edited for every new
    checkpoint; this one asks each file the question directly.
    """
    return any(str(label).startswith(("B-", "I-")) for label in labels)


def entity_spans(result):
    """`{(text, type)}` from a per-token BIO labelling, pieces glued back into words.

    The IOB2 convention the CoNLL family uses: `B-X` opens a span, `I-X` continues the open one, `O`
    closes it. A stray `I-X` with nothing open opens a span anyway rather than being dropped -- a
    model that emits one is doing something worth seeing in the failure message, not something to
    quietly normalise away.

    Pieces are joined bare because that is how the export writes them: `wordpiece_tokenizer_export`
    applies llama.cpp's `phantom()` transform, so a continuation piece has lost its "##" and a
    word-initial one carries the word boundary. Lowercased on the way out, since whether a checkpoint
    is cased is not what this is asking.
    """
    spans, current, current_type = set(), [], None

    def close():
        if current:
            spans.add(("".join(current).strip().lower(), current_type))

    for token in result:
        label = token.label or ""
        tag, _, kind = label.partition("-")
        if tag == "B" or (tag == "I" and kind != current_type):
            close()
            current, current_type = [token.piece], kind
        elif tag == "I":
            current.append(token.piece)
        else:
            close()
            current, current_type = [], None
    close()
    return spans


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_token_classification_finds_the_entities(name, jfk, tmp_path, monkeypatch):
    """A token classifier labels its own card's sentence with what is actually in it.

    TWO READINGS OF ONE DOOR, and which one applies is read off the file's own label set rather than
    off its name (`labels_are_spans`): a CoNLL head returns IOB2 spans and a punctuation head returns
    the mark that follows each token. Both come back through `text2class` in the identical shape, so a
    single expectation could only serve one of them -- and the family's third checkpoint (P5,
    2026-09-04) is the second kind.

    The *is it right* question for this family, and it needs its own answer for the reason the TTS row
    needed one: "the block ran" is satisfied by a model that returns `O` for every token, which is
    exactly what a broken export does -- a randomly-initialised head, a baked sequence length reached
    at the wrong length, a vocabulary whose ids do not match what the graph was trained on. All three
    produce a `Classification` of the right shape and the wrong contents.

    Graded on the card's OWN result, for the same reason the ASR row is: a call invented here would be
    a second, unpublished spelling of the thing under test. `produced` takes the LAST one the card
    bound, which for a card that demonstrates `strip_special=False` last is the unstripped labelling --
    which is fine and slightly stronger: the framing rows are graded too, so a model that invented an
    entity out of `[CLS]` would fail here. The strip itself is pinned hermetically, in loom.cpp's
    `tests/ci/test_text_classify.cpp`, where it does not need a real checkpoint.
    """
    _cards_dir()
    gguf, readme = _entry(name)
    if loom.contract_of(gguf).get("interface") != "text2class":
        pytest.skip(f"{name} is not text2class")

    ns, unmet = run_card(name, gguf, readme, jfk, tmp_path, monkeypatch)
    result = produced(ns, "tokens", "labels")
    if result is None:
        pytest.skip(f"{name}'s card labelled nothing{' -- ' + unmet if unmet else ''}")

    assert len(result), f"{name} labelled a sentence and produced no tokens"
    # The label SET is the file's, and a card that prints it is printing what the model can choose
    # between -- so an export that lost `loom.labels` shows up here rather than as bare integers in
    # somebody's terminal.
    assert result.labels, f"{name} declares no label names, so its ids mean nothing to a reader"
    assert all(t.label for t in result), (
        f"{name} returned a class id with no name: "
        f"{[(t.piece, t.label_id) for t in result if not t.label][:5]}"
    )

    if labels_are_spans(result.labels):
        found = entity_spans(result)
        assert found >= CLASSIFY_ENTITIES, (
            f"{name} did not find the entities in its own card's sentence.\n"
            f"  expected at least: {sorted(CLASSIFY_ENTITIES)}\n"
            f"  found:             {sorted(found)}\n"
            f"  labelling:         {[(t.piece, t.label) for t in result]}"
        )
    else:
        # A punctuation checkpoint. Same question -- did it get the sentence right -- against the
        # constant that means it here, and an all-`0` labelling (what a broken export produces) fails
        # this exactly as an all-`O` one fails the arm above.
        marked = {(t.piece.lower(), t.label) for t in result if t.label and t.label != "0"}
        assert marked >= CLASSIFY_MARKS, (
            f"{name} did not punctuate its own card's sentence.\n"
            f"  expected at least: {sorted(CLASSIFY_MARKS)}\n"
            f"  marked:            {sorted(marked)}\n"
            f"  labelling:         {[(t.piece, t.label) for t in result]}"
        )


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_audio_classifier_hears_the_reference(name, jfk, tmp_path, monkeypatch):
    """An audio classifier gives jfk.wav the answer it has: English, speech, rows spanning the clip.

    Family 13's *is it right* question, and it needs asking for the reason family 12's did: a dead
    head returns a well-formed `AudioClasses` -- right labels, right row count -- whose contents are
    one class everywhere. So the card's own result is graded against what the recording is, and
    which expectation applies is read off the LABELS the file declares rather than off its name: a
    `non_speech` class means a frame model whose other classes are speech, an `en: ...` label means a
    language id.

    Two answers are checked by a call made here rather than by the card, and both are about INPUT the
    card cannot ship: silence must come back as non-speech (a VAD that says "speech" to everything
    passes the speech-fraction check), and a frame model's row count must follow the clip length the
    harness knows.
    """
    _cards_dir()
    gguf, readme = _entry(name)
    if loom.contract_of(gguf).get("interface") != "speech2class":
        pytest.skip(f"{name} is not speech2class")

    ns, unmet = run_card(name, gguf, readme, jfk, tmp_path, monkeypatch)
    result = produced(ns, "probabilities", "labels", "granularity")
    if result is None:
        pytest.skip(f"{name}'s card classified nothing{' -- ' + unmet if unmet else ''}")

    assert result.labels, f"{name} declares no label names, so its rows mean nothing to a reader"
    assert len(result), f"{name} returned no rows"
    for row in result.probabilities:
        assert len(row) == len(result.labels), "every row is one probability per declared label"
        assert abs(sum(row) - 1.0) < 1e-3, f"{name} returned a row that is not a distribution: {sum(row)}"

    if result.granularity == "clip":
        assert len(result) == 1, f"a clip answer is one row, got {len(result)}"
        languages = [label for label in result.labels if ":" in label]
        if not any(label.startswith("en:") for label in languages):
            pytest.skip(f"{name} is a clip classifier with no English label; no expectation for jfk.wav")
        assert result.best[0].startswith("en:"), (
            f"{name} heard jfk.wav as {result.best[0]!r}; top 3: {result.top(3)}")
        return

    assert result.granularity == "frame", f"unknown granularity {result.granularity!r}"
    assert result.frame_rate > 0, f"{name} returned frame rows with no frame rate to place them in time"
    if "non_speech" not in result.labels:
        pytest.skip(f"{name} is a frame classifier with no non_speech class; no expectation for jfk.wav")
    silent = result.labels.index("non_speech")
    speech = sum(1 for row in result.probabilities if max(range(len(row)), key=row.__getitem__) != silent)
    assert speech / len(result) >= MIN_SPEECH_FRACTION, (
        f"{name} placed {speech} of {len(result)} jfk.wav frames in a speech class; a recording that "
        f"is mostly speech should get at least {MIN_SPEECH_FRACTION:.0%}")

    model = loom.Model.from_file(str(gguf))
    quiet = model.speech2class.infer([0.0] * (10 * 16000))
    expected = 10 * quiet.frame_rate
    assert abs(len(quiet) - expected) <= FRAME_ROW_SLACK, (
        f"{name} returned {len(quiet)} rows for 10 s at {quiet.frame_rate:.2f} frames/s; expected about "
        f"{expected:.0f}. A row count that does not follow the input is a baked length.")
    assert quiet.best.count("non_speech") == len(quiet), (
        f"{name} heard speech in silence: {len(quiet) - quiet.best.count('non_speech')} of {len(quiet)} frames")


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_audio_embedder_tells_speakers_apart(name, jfk, tmp_path, monkeypatch):
    """A speaker embedding scores one speaker as one speaker, and a different voice as different.

    The card is run first, and must bind a vector -- a list of floats -- that is finite and not zero.
    The comparison is made here, because the card can only ship one voice: the two halves of jfk.wav
    are the same speaker, and jfk.wav resampled 1.4x (higher and faster) is a different one to the
    model. A constant output, which is what a broken embedder produces, passes the first and fails the
    second.
    """
    _cards_dir()
    gguf, readme = _entry(name)
    contract = loom.contract_of(gguf)
    if contract.get("interface") != "speech2embeddings":
        pytest.skip(f"{name} is not speech2embeddings")
    if contract.get("output_granularity") != "clip":
        pytest.skip(f"{name}'s embeddings are per {contract.get('output_granularity')!r}, not per clip")

    ns, unmet = run_card(name, gguf, readme, jfk, tmp_path, monkeypatch)
    vectors = [v for v in ns.values()
               if isinstance(v, list) and len(v) >= 16 and all(isinstance(x, float) for x in v)]
    if not vectors:
        pytest.skip(f"{name}'s card embedded nothing{' -- ' + unmet if unmet else ''}")
    vector = vectors[-1]
    assert all(math.isfinite(x) for x in vector), f"{name} returned a non-finite embedding"
    assert any(x != 0.0 for x in vector), f"{name} returned an all-zero embedding"

    model = loom.Model.from_file(str(gguf))
    half = len(jfk) // 2
    first = model.speech2embeddings.infer(jfk[:half])
    second = model.speech2embeddings.infer(jfk[half:])
    other = model.speech2embeddings.infer([jfk[int(i * 1.4)] for i in range(int(len(jfk) / 1.4))])
    same, different = _cosine(first, second), _cosine(first, other)
    assert same >= MIN_SAME_SPEAKER, f"{name} scored two halves of one speaker at cosine {same:.2f}"
    assert different <= MAX_OTHER_SPEAKER, (
        f"{name} scored a different voice at cosine {different:.2f} (same speaker: {same:.2f})")


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_frame_features_follow_the_clip_in_every_precision(name, jfk, tmp_path, monkeypatch):
    """A frame-level feature extractor (embeddings per FRAME; the card calls the `speech2embeddings`
    door, which answers a `FrameEmbeddings`): one row per frame of the clip, every row the declared
    width, each row placed at the time the file's frame rate gives, and the rows MOVE with the audio.

    Then every GGUF the repo carries -- wakehubert-tiny ships four precisions -- is run on jfk.wav
    through the same door and compared with the largest frame by frame. The card's snippet loads one of
    them, and a release that only ever executed that one would publish three files nothing had run. The
    floor is generous on purpose (Q4_1 measured a mean cosine of 0.987 against PyTorch): it catches a
    file that is BROKEN -- the wrong weights, a scrambled layout, a dead layer -- not one that is merely
    coarser.
    """
    _cards_dir()
    gguf, readme = _entry(name)
    contract = loom.contract_of(gguf)
    if contract.get("output_kind") != "embeddings" or contract.get("output_granularity") != "frame":
        pytest.skip(f"{name} does not return embeddings per frame")
    hop = round(contract["sample_rate"] / contract["frame_rate"])
    width = contract.get("embedding_dim", 0)
    assert width > 0, (f"{name} returns embeddings per frame and declares no `embedding_dim`, so the "
                       f"door refuses it; re-export it with a current loom-exporter")

    ns, unmet = run_card(name, gguf, readme, jfk, tmp_path, monkeypatch)
    answers = [v for v in ns.values() if isinstance(v, loom.FrameEmbeddings)]
    if not answers:
        pytest.skip(f"{name}'s card produced no FrameEmbeddings{' -- ' + unmet if unmet else ''}")
    features = answers[-1]
    rows = features.rows
    assert len(rows) == len(jfk) // hop, (
        f"{name}'s card made {len(rows)} rows of jfk.wav; one per {hop} samples is {len(jfk) // hop}")
    assert features.dim == width and all(len(r) == width for r in rows), (
        f"{name}'s rows are not all the declared {width} wide")
    assert features.frame_rate == pytest.approx(contract["frame_rate"])
    assert features.times[1] - features.times[0] == pytest.approx(1 / contract["frame_rate"]), (
        f"{name}'s rows are not placed {1 / contract['frame_rate']:.4f} s apart")
    assert all(math.isfinite(x) for r in rows for x in r), f"{name} returned a non-finite feature"
    assert _cosine(rows[len(rows) // 4], rows[3 * len(rows) // 4]) < 0.99, (
        f"{name} returned near-identical features a quarter and three quarters into the clip")

    reference = loom.Model.from_file(str(gguf)).speech2embeddings.infer(jfk).rows
    for other in sorted(gguf.parent.glob("*.gguf")):
        other_rows = loom.Model.from_file(str(other)).speech2embeddings.infer(jfk).rows
        assert len(other_rows) == len(reference), (
            f"{other.name} returned {len(other_rows)} rows, not {len(reference)}")
        cosines = [_cosine(a, b) for a, b in zip(other_rows, reference)]
        mean = sum(cosines) / len(cosines)
        assert mean >= MIN_PRECISION_AGREEMENT, (
            f"{other.name} agrees with {gguf.name} at a mean per-frame cosine of {mean:.4f}")


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_codec_output_length_follows_the_input(name, jfk, tmp_path, monkeypatch):
    """A codec decoder returns `frames * hop` samples, for the frames its own card asked for.

    THE *IS IT RIGHT* QUESTION FOR THIS FAMILY IS THE LENGTH, and that is not a guess about what might
    break -- it is what DID break. The first working DAC export produced correct audio and returned one
    frame's worth of it for every input, because the exporter's shape walk gave up on the RVQ's
    rank-reducing slice and every transposed convolution was cropped to a literal. Nothing raised: the
    export succeeded, the file loaded, the driver returned floats. A gate that only asked "did the
    block run" would have shipped it.

    Derived from what the FILE declares -- `sample_rate / frame_rate` is the hop -- so this is one
    check for every codec rather than a table of per-model constants.
    """
    _cards_dir()
    gguf, readme = _entry(name)
    if loom.contract_of(gguf).get("interface") != "codes2speech":
        pytest.skip(f"{name} is not codes2speech")
    model = loom.Model.from_file(str(gguf))

    ns, unmet = run_card(name, gguf, readme, jfk, tmp_path, monkeypatch)
    audio = produced(ns, "samples", "sample_rate")
    if audio is None:
        pytest.skip(f"{name}'s card decoded nothing{' -- ' + unmet if unmet else ''}")

    rate = int(model.contract["sample_rate"])
    frame_rate = float(model.hparam("codec.frame_rate", "f32"))
    hop = rate / frame_rate
    frames = round(float(model.hparam("codec.frame_rate", "f32")))   # what the card decodes
    # Times the channels: a stereo codec (MOSS-Audio-Tokenizer) returns interleaved `L R L R`, so its
    # run is twice as long as its duration in samples -- which the file declares, like the hop.
    channels = int(model.contract.get("channels") or 1)
    assert audio.channels == channels, "the waveform must carry the channel count the file declares"
    expected = round(frames * hop) * channels
    assert len(audio.samples) == expected, (
        f"{name} decoded {frames} frames to {len(audio.samples)} floats; at {hop:.1f} samples per "
        f"frame that should be {expected}. A length that does not follow the input is the failure "
        f"this row exists for -- it produces a plausible file and the wrong duration."
    )
    assert audio.sample_rate == rate, "the waveform must carry the rate the file declares"
    # Not silence and not clipped. All-zero codes are a valid input and decode to a real (if dull)
    # signal; a decoder that returned zeros would pass the length check and be broken.
    peak = max(abs(s) for s in audio.samples)
    assert peak <= MAX_PEAK, f"{name} peak {peak:.4f} is clipped"


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_declared_greedy_decoding_is_reproducible(name):
    """A model whose export declares temperature 0 gives the same answer twice.

    Only where the FILE says so. `gemma-3-270m-it` declares temperature 1.0, top_k 64, top_p 0.95 --
    faithfully, from its own generation_config -- so it samples by design and asserting determinism
    on it would be asserting that the export is unfaithful.
    """
    _cards_dir()
    gguf, _ = _entry(name)
    if loom.contract_of(gguf).get("interface") != "text2text":
        pytest.skip(f"{name} is not text2text")
    model = loom.Model.from_file(str(gguf))
    # `hparam` raises when the key is absent, and absent means the export declared no sampling
    # defaults at all -- which is greedy. There is no `has_hparam`, so this is the probe.
    try:
        temp = float(model.hparam("sampling.temperature", "f32"))
    except Exception:
        temp = 0.0
    if temp:
        pytest.skip(f"{name} declares temperature {temp}, so it samples by design")
    prompt = "The capital of France is"
    first = model.text2text.infer(prompt, max_new_tokens=12)
    second = model.text2text.infer(prompt, max_new_tokens=12)
    assert first == second, f"{name} declares greedy decoding but gave two answers:\n  {first!r}\n  {second!r}"


@pytest.mark.gate
@pytest.mark.parametrize("name", NAMES)
def test_a_codec_that_draws_its_own_noise_still_answers_the_same_twice(name):
    """A stochastic decoder must be reproducible by default, and must still be stochastic on request.

    SNAC's decoder is the first graph here with a random leaf: its driver draws the noise per call
    through `loom.gaussian_array`, which is a fresh waveform every call unless the stream is seeded.
    It IS seeded -- `inputs.seed`, defaulting to a fixed value -- and that default is what lets every
    other row in this file compare two runs of a codec at all. A build that dropped the seeding would
    fail nothing else here: the audio stays correct, the length stays right, and only a comparison
    between two calls can see it.

    The second half matters as much as the first. A file that answered the same twice because the
    noise was ignored -- the shape this export shipped with once, and had to reverse -- passes the
    determinism check perfectly.
    """
    _cards_dir()
    gguf, _ = _entry(name)
    if loom.contract_of(gguf).get("interface") != "codes2speech":
        pytest.skip(f"{name} is not codes2speech")
    model = loom.Model.from_file(str(gguf))
    width = model.hparam("codec.n_codebooks", "u32")
    frames = round(float(model.hparam("codec.frame_rate", "f32")))
    codes = [[(i * width + k) % 512 for k in range(width)] for i in range(frames)]

    first = model.codes2speech.infer(codes).samples
    second = model.codes2speech.infer(codes).samples
    assert list(first) == list(second), (
        f"{name} gave two different waveforms for one input -- an unseeded draw somewhere in its "
        f"driver, which makes every other comparison in this file meaningless"
    )
    seeded = model.codes2speech.infer(codes, seed=99).samples
    if list(seeded) == list(first):
        pytest.skip(f"{name} has no stochastic leaf -- `seed` changes nothing, which is correct for a "
                    f"deterministic codec like DAC")
    assert len(seeded) == len(first), "a seed must change the draw, not the length"


def _mono(audio):
    """The waveform as one channel, for the oracle: interleaved channels averaged. MOSS-Audio-Tokenizer
    is the first stereo output here, and a recogniser handed `L R L R` as one channel hears it at
    half speed."""
    samples = list(audio.samples)
    channels = int(getattr(audio, "channels", 1) or 1)
    if channels == 1:
        return samples
    return [sum(samples[i:i + channels]) / channels for i in range(0, len(samples), channels)]


def _resample_16k(samples, rate):
    """Linear resample to 16 kHz. An oracle, not a codec -- good enough to recognise words by."""
    if rate == 16000:
        return list(samples)
    n = int(round(len(samples) * 16000 / rate))
    out, step = [], (len(samples) - 1) / max(n - 1, 1)
    for i in range(n):
        x = i * step
        lo = int(x)
        hi = min(lo + 1, len(samples) - 1)
        out.append(samples[lo] + (samples[hi] - samples[lo]) * (x - lo))
    return out
