"""The binding against a real exported model.

Everything the CI suite checks is about the Python layer's own decisions. What it cannot check is the
claim this package actually makes: that loading a loom GGUF registers whatever topologies it declares,
attaches a cache to the ones that ask for it, and runs the driver the file shipped with -- for a model
this package has never heard of.

So this needs a real artifact, and skips without one:

    export LOOM_TEST_MODEL=~/loom-fixtures/matcha_mil.gguf
    pytest tests/gate -q

Deliberately written against no particular architecture. It asserts the *shape* of what a loom model
is, which is the whole of what this package knows -- a test that expected Matcha's inputs would be
this package learning about a model, which is the thing the design is supposed to make unnecessary.
"""
import os
from pathlib import Path

import pytest

import loom


@pytest.fixture(scope="module")
def model():
    path = os.environ.get("LOOM_TEST_MODEL")
    if not path:
        pytest.skip("LOOM_TEST_MODEL is not set; it names any loom GGUF to drive")
    if not Path(path).is_file():
        pytest.skip(f"{path} does not exist")
    return loom.Model.from_file(path)


def test_a_model_describes_itself(model):
    """The three things every loom GGUF carries, and the reason no per-architecture code is needed."""
    assert model.architecture
    assert model.topologies, "a loom GGUF declares at least one graph topology"
    assert all(isinstance(name, str) and name for name in model.topologies)
    assert isinstance(model.driver_source, str)
    assert model.has_driver == bool(model.driver_source)


def test_the_repr_is_useful_at_a_prompt(model):
    text = repr(model)
    assert model.architecture in text
    assert model.path.name in text


def test_a_wrong_hparam_kind_raises_rather_than_guessing(model):
    with pytest.raises(ValueError, match="unknown hparam kind"):
        model.hparam("anything", kind="int64")


def test_calling_a_function_the_driver_does_not_define(model):
    """The engine's own error type, surfaced as itself."""
    if not model.has_driver:
        pytest.skip("this artifact carries no driver")
    with pytest.raises(loom.LoomError, match="no such Lua function"):
        model.call("not_a_real_entry_point", {})


def test_inference_runs_and_is_deterministic(model):
    """`LOOM_TEST_MODEL_INPUTS` supplies the driver's own arguments as JSON, because which arguments a
    driver takes is a property of the model. Without it this test cannot know what to pass and says
    so, rather than guessing a signature and reporting the failure as a bug in the binding."""
    import json

    raw = os.environ.get("LOOM_TEST_MODEL_INPUTS")
    if not raw:
        pytest.skip("set LOOM_TEST_MODEL_INPUTS to a JSON object of the driver's inputs "
                    "(model.driver_source documents them)")
    inputs = json.loads(raw)

    first = model.infer(**inputs)
    assert isinstance(first, (list, float))
    if isinstance(first, list):
        assert first, "the driver returned an empty result"
        assert all(isinstance(x, float) for x in first)

    # A loom driver is deterministic given its inputs -- any randomness is seeded through them -- so
    # two identical calls must agree. This also exercises the second call against a cache and graph
    # buffers the first one left behind, which is where a stateful engine gets caught.
    assert model.infer(**inputs) == first


def test_a_text_model_goes_from_a_string_to_a_string(model):
    """The end-to-end claim, on whatever real model is configured.

    Skips for a model with no embedded vocabulary -- which is not a gap in the model but a fact about
    it: the TTS families take phoneme ids a phonemiser produces outside the engine, so there is no
    text door to test.
    """
    if model.tokenizer is None:
        pytest.skip(f"{model.architecture} embeds no vocabulary; its driver takes ids directly")

    ids = model.tokenize("The capital of France is")
    assert ids and all(isinstance(i, int) for i in ids)
    # Round-tripping is not required to be the identity -- a tokenizer may normalise, and some add a
    # BOS -- but the text must survive it.
    assert "capital of France" in model.detokenize(ids)

    out = model.generate("The capital of France is", max_new_tokens=8)
    assert isinstance(out, str) and out.strip(), "generate produced nothing"


def test_the_tokenizer_describes_itself(model):
    if model.tokenizer is None:
        pytest.skip(f"{model.architecture} embeds no vocabulary")
    tokenizer = model.tokenizer
    assert tokenizer.kind in {"gpt2", "bert", "byt5", "llama", "t5", "ctc", "phonemes",
                              "supertonic"}, tokenizer.kind
    assert tokenizer.size > 0


def test_a_live_stream_transcribes_what_one_pass_does(model):
    """The live door (loom.cpp ADR-098), on whatever streaming ASR model is configured: the same
    transcript as `transcribe`, however the audio is cut into pieces, with two streams open at once.
    Skips for a file with no live path -- every file exported before streaming."""
    if not model.can_stream:
        pytest.skip(f"{model.architecture} has no live path (stream_open / stream_feed / stream_finish)")
    import math
    import wave
    rate = int(model.contract.get("sample_rate") or 16000)
    if os.environ.get("LOOM_TEST_WAV"):
        # Real speech, 16-bit mono at the model's rate: a transcript long enough to cross many chunks.
        with wave.open(os.environ["LOOM_TEST_WAV"]) as w:
            assert w.getframerate() == rate and w.getnchannels() == 1 and w.getsampwidth() == 2
            raw = w.readframes(w.getnframes())
        clip = [int.from_bytes(raw[i:i + 2], "little", signed=True) / 32768.0 for i in range(0, len(raw), 2)]
    else:
        # A rising tone with an envelope: not speech, but a deterministic input the encoder answers.
        clip = [0.3 * math.sin(2 * math.pi * (200 + 400 * t / (3 * rate)) * t / rate)
                * math.sin(math.pi * t / (3 * rate)) for t in range(3 * rate)]
    want = model.transcribe(clip).text
    assert want, "an empty transcript would make the comparison below vacuous"
    a, b = model.speech2text.stream(), model.speech2text.stream()
    for at in range(0, len(clip), 1999):
        a.push(clip[at:at + 1999])
        if at % (2 * 1999) == 0:
            b.push(clip[at:at + 2 * 1999])
    a.finish()
    b.finish()
    assert a.text == want and b.text == want


def test_a_token_stream_is_what_generate_returns(model):
    """The tokens-out door (loom.cpp ADR-099), on whatever LM is configured: the pieces join to
    `generate`'s text, greedy and seeded, and two streams pulled in turn -- with a one-pass `generate`
    between their pulls -- each give their own, under both cache modes. Skips for a model that is not
    text2text, or a file whose loop has no stream functions."""
    if not model.text2text.supported or not model.can_stream_tokens:
        pytest.skip(f"{model.architecture} has no token stream")
    prompt_a, prompt_b = "The history of the city of Rome begins", "A recipe for bread needs"
    want_a = model.generate(prompt_a, max_new_tokens=24, temperature=0.0)
    want_b = model.generate(prompt_b, max_new_tokens=24, temperature=0.8, top_k=40, seed=7)
    assert want_a, "an empty generation would make the comparison below vacuous"
    assert "".join(model.text2text.stream(prompt_a, max_new_tokens=24, temperature=0.0)) == want_a
    for mode in ("park", "per-stream"):
        model.stream_caches = mode
        a = iter(model.generate_stream(prompt_a, max_new_tokens=24, temperature=0.0, per_pull=3))
        b = iter(model.generate_stream(prompt_b, max_new_tokens=24, temperature=0.8, top_k=40, seed=7))
        got_a, got_b = [], []
        for _ in range(64):
            got_a.append(next(a, ""))
            assert model.generate("The three primary colours are", max_new_tokens=4, temperature=0.0)
            got_b.append(next(b, ""))
        assert "".join(got_a) == want_a, mode
        assert "".join(got_b) == want_b, mode
    model.stream_caches = "auto"


def test_a_speech_stream_is_what_infer_returns(model):
    """The audio-out door (loom.cpp ADR-100), on whatever TTS is configured: a file whose driver streams
    hands back pieces that join to `infer`'s waveform sample for sample, and two streams pulled in turn
    -- with a one-pass `infer` between their pieces -- each give their own. Skips for a model that is not
    text2speech, or a file whose driver does not stream (that one is cut by sentence, not exactly)."""
    if not model.text2speech.supported or not model.can_stream_speech:
        pytest.skip(f"{model.architecture} has no speech stream")
    text_a = ("The history of the city of Rome begins on the Palatine hill, where shepherds built their "
              "first huts beside the river. A good recipe for country bread needs only flour, water, salt "
              "and a great deal of patience in the morning.")
    text_b = "Trains leave every twenty minutes. Bring a warm coat."
    want_a = model.text2speech.infer(text_a, seed=11).samples
    want_b = model.text2speech.infer(text_b, seed=12).samples
    stream = model.text2speech.stream(text_a, seed=11)
    assert stream.exact
    pieces = [p.samples for p in stream]
    assert len(pieces) >= 2 and sum(pieces, []) == want_a
    a = iter(model.text2speech.stream(text_a, seed=11))
    b = iter(model.text2speech.stream(text_b, seed=12))
    got_a, got_b = [], []
    for _ in range(16):
        got_a += next(a, loom.Audio(samples=[], sample_rate=1)).samples
        assert model.text2speech.infer(text_b, seed=12).samples == want_b
        got_b += next(b, loom.Audio(samples=[], sample_rate=1)).samples
    assert got_a == want_a and got_b == want_b
