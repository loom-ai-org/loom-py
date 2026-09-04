"""The family-10 pair driven through the doors, against the real GGUFs: text -> codes -> waveform.

**Why this is a test at all.** Dia emits codec tokens and no audio; DAC turns codec tokens into audio.
loom.cpp's ADR-022 decided they stay two files -- one codec serves ~20 autoregressive LMs, and the
codes are the useful intermediate -- and the cost of that decision is exactly this: with two files,
nothing inside either one asserts that they fit together. The engine-side version of this check is
loom.cpp's `test_e2e_dia_dac_composition.cpp`, which compares both halves against `transformers`. What
is left for HERE is the part that is this package's own: that the composition a user writes,

    codes = dia.text2codes.infer("[S1] Hello world.")
    audio = dac.codes2speech.infer(codes)

is two calls and the array between them, with no reshaping, no width argument and no delay
bookkeeping in the middle.

**Running it:**

    export LOOM_DIA_MIL_GGUF=$LOOM_FIXTURES/dia_mil.gguf
    export LOOM_DAC_44KHZ_GGUF=$LOOM_FIXTURES/dac_44khz.gguf
    pytest tests/gate/test_codec_pair.py -q

Both are the same variables loom.cpp's gate suite reads, so one export serves both repos. It skips
cleanly without them, like every gate test here.

**It generates eight frames, and that is deliberate.** This is a shape-and-composition check, not a
quality one: the numbers are pinned against `transformers` on the engine side, and the *is it right*
question for this family -- does it sound like the words -- cannot be asked of a driver that is still
greedy and classifier-free-guidance-free (loom.cpp Retro-006 on why correlation is not that test).
Eight frames is 93 ms of audio and about twenty decoder steps of a 1.6 B model, which is what makes
this affordable to run beside the rest of the gate suite.
"""
import os
from pathlib import Path

import pytest

import loom

FRAMES = 8


def _model(var: str):
    path = os.environ.get(var)
    if not path:
        pytest.skip(f"{var} is not set")
    if not Path(path).is_file():
        pytest.skip(f"{path} does not exist")
    return loom.Model.from_file(path)


@pytest.fixture(scope="module")
def lm():
    """The AR codec LM. Module-scoped because it is 6.4 GB of F32 weights and every test here wants
    the same one -- a per-test fixture would load it four times."""
    return _model("LOOM_DIA_MIL_GGUF")


@pytest.fixture(scope="module")
def codec():
    return _model("LOOM_DAC_44KHZ_GGUF")


@pytest.fixture(scope="module")
def codes(lm):
    """One generation, shared: it is the input to everything below and costs a real decode loop."""
    # **Greedy and guidance-free, named rather than defaulted.** This file declares the checkpoint's
    # own decoding -- sampling at 1.8/50/0.9 with classifier-free guidance at 3.0 -- so an `infer`
    # that named neither would draw a different generation every run and pay two decoder passes per
    # step for it. Everything asserted here is about the SHAPE of the composition, and the cheapest
    # real generation is the right input for that. Whether the guided decode is right is asked where
    # a difference is attributable: loom.cpp's `test_e2e_dia_mil_export.cpp`, on the codes.
    return lm.text2codes.infer("[S1] Hello world.", max_new_tokens=FRAMES,
                               temperature=0.0, guidance_scale=1.0)


def test_the_lm_declares_the_codes_door_and_not_the_audio_one(lm):
    """`audio_codes` as an OUTPUT kind is what separates these two families. A file that declared
    `audio` here would resolve to `text2speech` and be handed a door it cannot answer -- which is the
    fold loom.cpp's ADR-020 exists to prevent."""
    assert lm.capabilities == ("text2codes",)
    assert not lm.text2speech.supported
    assert lm.task == "text-to-codes"


def test_the_codec_declares_the_other_half(codec):
    assert codec.capabilities == ("codes2speech",)
    assert codec.task == "audio-codec"


def test_the_two_files_agree_on_the_width_of_a_frame(lm, codec):
    """`loom.codec.n_codebooks`, written on both sides under the same key -- the LM from its channel
    count, the codec from its quantizer count. It is the one fact a host must check before chaining
    two files it did not export together, and it is checkable without running either of them."""
    assert lm.hparam("codec.n_codebooks", "u32") == codec.hparam("codec.n_codebooks", "u32")


def test_the_door_returns_frames_of_the_declared_width(lm, codes):
    """Frame-major rows, `n_codebooks` wide, `max_new_tokens` of them.

    `max_new_tokens` counting AUDIO FRAMES rather than decoder rows is the interface's own choice and
    worth pinning: rows differ from frames by the delay pattern, which is an artefact of how an AR
    codec LM writes its codebooks rather than anything a caller asked for."""
    width = lm.hparam("codec.n_codebooks", "u32")
    assert len(codes) == FRAMES
    assert all(len(row) == width for row in codes)
    assert all(isinstance(c, int) for row in codes for c in row)
    # Not one value repeated: a delay scaffold that never got undone returns the BOS id everywhere,
    # which has the right shape and no audio in it.
    assert len({tuple(row) for row in codes}) > 1


def test_the_codes_go_straight_into_the_codec(lm, codec, codes):
    """The composition, with nothing between the two calls. That is the assertion -- a pair that
    needed a transpose or a width argument here would still pass every test above."""
    audio = codec.codes2speech.infer(codes)
    rate = codec.contract["sample_rate"]
    hop = round(rate / codec.hparam("codec.frame_rate", "f32"))
    assert audio.sample_rate == rate
    assert len(audio.samples) == FRAMES * hop, (
        "a codec decoder's answer is its LENGTH -- the first working DAC export returned one frame's "
        "worth of audio for every input and raised nothing"
    )
    peak = max(abs(s) for s in audio.samples)
    assert 0.0 < peak <= 1.0, "silence and clipping are both plausible-looking failures here"
