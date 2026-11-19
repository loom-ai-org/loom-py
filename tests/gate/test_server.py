"""`loom-serve` with real models, through OpenAI's own client (loom-py#52).

What `tests/ci/test_server.py` cannot show: that the official `openai` package reads every answer --
its SSE parser, its binary responses, its multipart upload -- and that the doors underneath give the
same text a direct call does. Each door skips without its model:

    LOOM_SERVE_CHAT=smollm2-360m-instruct.gguf      # text2text with a chat template
    LOOM_SERVE_STREAM=lfm2-350m-modular.gguf        # text2text whose file streams tokens
    LOOM_SERVE_ASR=moonshine-tiny-q8_0.gguf         # speech2text
    LOOM_SERVE_TTS=paradee-8m-q8_0_pc.gguf          # text2speech with built-in voices
    pip install openai && pytest tests/gate/test_server.py -q
"""
import io
import os
import threading
import wave
from pathlib import Path

import pytest

import loom
from loom import server as srv

openai = pytest.importorskip("openai")

JFK_WAV = Path(__file__).resolve().parents[2] / "vendor" / "loom.cpp" / "samples" / "jfk.wav"
DOORS = {"chat": "LOOM_SERVE_CHAT", "stream": "LOOM_SERVE_STREAM", "asr": "LOOM_SERVE_ASR", "tts": "LOOM_SERVE_TTS"}


@pytest.fixture(scope="module")
def served():
    models = {name: loom.Model.from_file(os.environ[env]) for name, env in DOORS.items()
              if os.environ.get(env) and Path(os.environ[env]).is_file()}
    if not models:
        pytest.skip("set one of " + ", ".join(DOORS.values()))
    s = srv.make_server(models, port=0, quiet=True)
    threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    client = openai.OpenAI(base_url=f"http://127.0.0.1:{s.server_address[1]}/v1", api_key="unused")
    yield models, client
    s.shutdown()
    s.server_close()


def need(served, name):
    if name not in served[0]:
        pytest.skip(f"{DOORS[name]} is not set")
    return served[0][name], served[1]


def test_chat_is_the_models_own_chat(served):
    model, client = need(served, "chat")
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    r = client.chat.completions.create(model="chat", messages=messages, max_tokens=24, temperature=0)
    assert r.choices[0].message.content == model.chat(messages, max_new_tokens=24, temperature=0)
    assert r.usage.prompt_tokens == len(model.tokenize(model.apply_chat_template(messages)))
    short = client.chat.completions.create(model="chat", messages=messages, max_tokens=2, temperature=0)
    assert short.choices[0].finish_reason == "length" and short.usage.completion_tokens == 2


def test_a_stream_joins_to_the_whole_reply(served):
    model, client = need(served, "stream")
    messages = [{"role": "user", "content": "Name three colours."}]
    pieces, finish = [], None
    for chunk in client.chat.completions.create(model="stream", messages=messages, max_tokens=32,
                                                temperature=0, stream=True):
        if chunk.choices and chunk.choices[0].delta.content:
            pieces.append(chunk.choices[0].delta.content)
        if chunk.choices and chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
    assert len(pieces) > 1, "a file that streams hands the reply back in pieces"
    assert "".join(pieces) == model.chat(messages, max_new_tokens=32, temperature=0)
    assert finish in ("stop", "length")


def test_transcription_is_the_models_own(served):
    model, client = need(served, "asr")
    with open(JFK_WAV, "rb") as f:
        r = client.audio.transcriptions.create(model="whisper-1", file=f)
    with wave.open(str(JFK_WAV)) as w:
        raw = w.readframes(w.getnframes())
    samples = [int.from_bytes(raw[i:i + 2], "little", signed=True) / 32768.0 for i in range(0, len(raw), 2)]
    assert r.text == model.transcribe(samples).text
    assert "ask not what your country can do for you" in r.text.lower()


def test_speech_in_each_format(served):
    model, client = need(served, "tts")
    voice = model.voices[0] if model.voices else "alloy"
    wav = client.audio.speech.create(model="tts-1", voice=voice, input="Hello from loom.", response_format="wav")
    with wave.open(io.BytesIO(wav.content)) as w:
        assert w.getframerate() == model.contract["sample_rate"] and w.getnframes() > w.getframerate() // 4
    with client.audio.speech.with_streaming_response.create(model="tts-1", voice=voice, input="One. Two.",
                                                            response_format="pcm") as resp:
        assert sum(len(b) for b in resp.iter_bytes()) > 0
    if srv._ffmpeg():
        mp3 = client.audio.speech.create(model="tts-1", voice=voice, input="Hello.")
        assert mp3.content[:3] == b"ID3" or mp3.content[:2] == b"\xff\xfb"
