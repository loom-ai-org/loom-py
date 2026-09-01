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
executed in order, in one namespace, exactly as a reader would follow them top to bottom. Precisely
ONE substitution is made:

    loom.Model.from_pretrained("loom-ai-org/<repo>")  ->  loom.Model.from_file("<local .gguf>")

because a release gate must test the artefact about to be published, not the one already on the Hub.
Everything else runs as printed. A card that needs an argument it does not show, or shows one the API
no longer takes, fails here.

**Three questions, per the family:**

* *does it work* -- the block runs to completion;
* *is it consistent* -- a model whose export declares greedy decoding gives the same answer twice;
* *is it right* -- and for a TTS model this is the only question that matters, because correlation is
  not the test. Kokoro once matched PyTorch at cosine 0.996 and shipped unintelligible
  (loom.cpp Retro-006), so synthesised audio is TRANSCRIBED BACK by a standard ASR model and
  compared with the words the card asked for. All five TTS cards say "hello world", which also makes
  them comparable with each other.

**Running it:**

    export LOOM_MODEL_CARDS=~/Dev/loom/hf-models      # the staging tree, cards beside GGUFs
    pytest tests/gate/test_model_cards.py -q

It skips cleanly without that variable, like every gate test. `pip install "loom-py-rt[phonemes]"`
additionally covers the text-in door; without it the cards' G2P lines are reported as skipped
preconditions rather than failures, because a missing optional extra is not a broken card.
"""
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

# The model that reads TTS output back. Whisper rather than a NeMo model because it is the one every
# card set already depends on for the ASR examples, and because its own card is checked here too --
# a broken oracle would fail its own row first, which is the ordering you want.
ORACLE = "whisper-small"

# PER-MODEL BASELINES, MEASURED, because one global ceiling cannot serve this family. Five of the
# seven transcribe jfk.wav PERFECTLY; GigaAM scores 0.50 and is not broken. A ceiling loose enough
# for GigaAM (0.6) would let Whisper rot from 0.00 to 0.30 unnoticed, and a tight one fails a model
# that works. What this gate is for is BREAKAGE, not quality -- a broken model scores about 1.0
# (silence, or noise) -- and breakage is per-model distance from where that model actually sits.
#
# Measured 2026-08-31 against samples/jfk.wav, on the rc7 exports:
ASR_BASELINE = {
    "conformer-ctc-small":    0.00,
    "granite-speech-4.0-1b":  0.00,
    "parakeet-rnnt-0.6b":     0.00,
    "parakeet-tdt-0.6b":      0.00,
    "whisper-small":          0.00,
    # Russian-first (it declares `ru, en`), so English costs it. The transcript is unmistakably the
    # right utterance -- "my fellow americans ... your country can do for you" -- spelled through a
    # recogniser trained elsewhere. Not a defect, and not something to tighten.
    "gigaam-v3-rnnt":         0.50,
    # ENTIRELY THE CONTROL MARKERS, and this number is a to-do rather than a property of the model.
    # Its speech transcription is perfect; the 0.18 is `language English<asr_text>` being counted as
    # four spurious words against a 21-word reference. Granite, the other family-3 model, comes back
    # clean, so this is one model's prompt scaffolding leaking into its output. When that is fixed
    # this baseline should drop to 0.00 -- and this gate will say so by passing with room to spare.
    "qwen3-asr-0.6b":         0.18,
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
    """
    return re.sub(
        r"loom\.Model\.from_pretrained\(\s*['\"][^'\"]+['\"]\s*\)",
        f"loom.Model.from_file({str(gguf)!r})",
        block,
    )


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
    ns = {"loom": loom, "audio": jfk}
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
    if loom.Model.from_file(str(gguf)).contract.get("interface") != "speech2text":
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
    if loom.Model.from_file(str(gguf)).contract.get("interface") != "text2speech":
        pytest.skip(f"{name} is not text2speech")

    ns, unmet = run_card(name, gguf, readme, jfk, tmp_path, monkeypatch)
    # By shape, and the shape matters: `audio` is SEEDED with the reference recording, which is a
    # plain list. Only a synthesised waveform carries `sample_rate`, so a TTS card that never
    # produced one cannot slip through with this test grading the oracle on jfk.wav and passing.
    audio = produced(ns, "samples", "sample_rate")
    if audio is None:
        pytest.skip(f"{name}'s card synthesised nothing{' -- ' + unmet if unmet else ''}")
    samples = list(audio.samples)
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
def test_declared_greedy_decoding_is_reproducible(name):
    """A model whose export declares temperature 0 gives the same answer twice.

    Only where the FILE says so. `gemma-3-270m-it` declares temperature 1.0, top_k 64, top_p 0.95 --
    faithfully, from its own generation_config -- so it samples by design and asserting determinism
    on it would be asserting that the export is unfaithful.
    """
    _cards_dir()
    gguf, _ = _entry(name)
    model = loom.Model.from_file(str(gguf))
    if model.contract.get("interface") != "text2text":
        pytest.skip(f"{name} is not text2text")
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
