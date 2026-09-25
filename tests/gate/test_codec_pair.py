"""The family-10 pairs driven through the doors, against the real GGUFs: text -> codes -> waveform.

**Why this is a test at all.** Dia emits codec tokens and no audio; DAC turns codec tokens into audio.
loom.cpp's ADR-022 decided they stay two files -- one codec serves ~20 autoregressive LMs, and the
codes are the useful intermediate -- and the cost of that decision is exactly this: with two files,
nothing inside either one asserts that they fit together. The engine-side versions of this check are
loom.cpp's `test_e2e_dia_dac_composition.cpp` and `test_e2e_moss_tts_composition.cpp`, which compare
both halves against the reference implementations. What is left for HERE is the part that is this
package's own: that the composition a user writes,

    codes = lm.text2codes.infer("...")
    audio = codec.codes2speech.infer(codes)

is two calls and the array between them, with no reshaping, no width argument and no delay
bookkeeping in the middle.

**Two pairs, and the second is not the first again.** MOSS-TTS emits 12 codebooks into
MOSS-Audio-Tokenizer's 32: the codec is a residual quantizer, declares the id that means "this codebook
is absent", and `codes2speech` fills each narrower row with it (loom.cpp ADR-050). So for this pair
"the files agree on the width" is "the LM is no wider, and the codec says how to fill the rest" -- and
the codec returns interleaved STEREO, so its length is frames x hop x channels.

**Running it:**

    export LOOM_DIA_MIL_GGUF=$LOOM_FIXTURES/dia_mil.gguf
    export LOOM_DAC_44KHZ_GGUF=$LOOM_FIXTURES/dac_44khz.gguf
    export LOOM_MOSS_TTS_GGUF=$LOOM_FIXTURES/moss_tts.gguf
    export LOOM_MOSS_AUDIO_TOKENIZER_GGUF=$LOOM_FIXTURES/moss_audio_tokenizer.gguf
    pytest tests/gate/test_codec_pair.py -q

They are the same variables loom.cpp's gate suite reads, so one export serves both repos. A pair whose
files are not set skips cleanly, like every gate test here. The MOSS pair is 21 GB at F32; the pairs
load one at a time, each torn down before the next.

**It generates eight frames, and that is deliberate.** This is a shape-and-composition check, not a
quality one: the numbers are pinned against the references on the engine side, and the *is it right*
question for this family -- does it sound like the words -- is the model-card gate's, which runs the
published cards through an ASR oracle. Eight greedy frames is the cheapest real generation.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import loom

FRAMES = 8


@dataclass(frozen=True)
class Pair:
    name: str
    lm_var: str
    codec_var: str
    text: str
    # The decode that makes a generation cheap and repeatable, named rather than defaulted: both
    # files declare their checkpoints' own sampling, so an `infer` that named nothing would draw a
    # different take every run.
    greedy: dict = field(default_factory=dict)


PAIRS = [
    # Dia: greedy and guidance-free. This file declares sampling at 1.8/50/0.9 with classifier-free
    # guidance at 3.0; the guided decode is checked where a difference is attributable, loom.cpp's
    # `test_e2e_dia_mil_export.cpp`, on the codes.
    Pair("dia-dac", "LOOM_DIA_MIL_GGUF", "LOOM_DAC_44KHZ_GGUF", "[S1] Hello world.",
         dict(temperature=0.0, guidance_scale=1.0)),
    # MOSS: greedy for BOTH draws -- the continue/stop head and the codebooks -- and a language, which
    # is the one argument this LM takes that Dia does not (loom.cpp ADR-052). The sentence runs for 38
    # frames under greedy, so eight never meets its stop.
    Pair("moss", "LOOM_MOSS_TTS_GGUF", "LOOM_MOSS_AUDIO_TOKENIZER_GGUF",
         "The quick brown fox jumps over the lazy dog.",
         dict(temperature=0.0, text_temperature=0.0, language="en")),
]


def _model(var: str):
    path = os.environ.get(var)
    if not path:
        pytest.skip(f"{var} is not set")
    if not Path(path).is_file():
        pytest.skip(f"{path} does not exist")
    return loom.Model.from_file(path)


@pytest.fixture(scope="module", params=PAIRS, ids=lambda p: p.name)
def pair(request):
    return request.param


@pytest.fixture(scope="module")
def lm(pair):
    """The AR codec LM. Module-scoped because it is gigabytes of F32 weights (6.4 GB for Dia, 16.8 GB
    for MOSS-TTS) and every test here wants the same one; pytest tears a pair down before setting up
    the next, and a dropped Model frees its weights at once (loom.cpp Retro-061)."""
    return _model(pair.lm_var)


@pytest.fixture(scope="module")
def codec(pair):
    return _model(pair.codec_var)


@pytest.fixture(scope="module")
def codes(pair, lm):
    """One generation per pair, shared: it is the input to everything below and costs a real decode
    loop."""
    return lm.text2codes.infer(pair.text, max_new_tokens=FRAMES, **pair.greedy)


def _absent_code(codec):
    try:
        return int(codec.hparam("codec.absent_code", "u32"))
    except Exception:
        return None


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
    two files it did not export together, and it is checkable without running either of them.

    Equal, or the LM NARROWER and the codec declaring the id that fills the rest: a residual codec
    decodes a prefix of its codebooks (MOSS, ADR-050). A narrower LM beside a codec that declares no
    such id is a pair that does not fit, and this is where that has to show."""
    lm_width = lm.hparam("codec.n_codebooks", "u32")
    codec_width = codec.hparam("codec.n_codebooks", "u32")
    if lm_width == codec_width:
        return
    assert lm_width < codec_width, f"the LM emits {lm_width} codebooks into a codec of {codec_width}"
    assert _absent_code(codec) is not None, (
        f"the LM emits {lm_width} of the codec's {codec_width} codebooks, and the codec declares no "
        f"`codec.absent_code` to fill the rest with"
    )


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
    needed a transpose, a width argument or a hand-padded row here would still pass every test
    above."""
    audio = codec.codes2speech.infer(codes)
    rate = codec.contract["sample_rate"]
    channels = int(codec.contract.get("channels") or 1)
    hop = round(rate / codec.hparam("codec.frame_rate", "f32"))
    assert audio.sample_rate == rate
    assert audio.channels == channels
    assert len(audio.samples) == FRAMES * hop * channels, (
        "a codec decoder's answer is its LENGTH -- the first working DAC export returned one frame's "
        "worth of audio for every input and raised nothing"
    )
    assert audio.duration == pytest.approx(FRAMES * hop / rate)
    peak = max(abs(s) for s in audio.samples)
    assert 0.0 < peak <= 1.0, "silence and clipping are both plausible-looking failures here"


def test_a_narrow_row_decodes_as_the_prefix_the_codec_declares(lm, codec, codes):
    """Only for a pair whose widths differ: the door's padding is the codec's absent id and nothing
    else. Padding by hand with that id must give the same waveform bit for bit, and padding with a
    real code (0) must not -- which is what makes the first comparison able to fail."""
    codec_width = codec.hparam("codec.n_codebooks", "u32")
    if len(codes[0]) == codec_width:
        pytest.skip("this pair's widths are equal; there is no prefix to decode")
    absent = _absent_code(codec)
    by_door = list(codec.codes2speech.infer(codes).samples)
    by_hand = list(codec.codes2speech.infer(
        [row + [absent] * (codec_width - len(row)) for row in codes]).samples)
    assert by_door == by_hand
    with_zeros = list(codec.codes2speech.infer(
        [row + [0] * (codec_width - len(row)) for row in codes]).samples)
    assert max(abs(a - b) for a, b in zip(by_door, with_zeros)) > 1e-3, (
        "filling with a real code decoded the same as filling with the absent id, so the comparison "
        "above could not have failed"
    )
