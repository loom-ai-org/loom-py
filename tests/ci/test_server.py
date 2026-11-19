"""`loom-serve` (loom/server.py, loom-py#52): OpenAI's HTTP API over the doors a model declares.

No GGUF: the models here are stand-ins with the methods the server calls, so what is pinned is the
server's half -- which model answers, how a request becomes a door call, and the bytes that come back
in OpenAI's shapes (JSON, server-sent events, WAV, PCM, subtitles, error objects). The half with real
models and the official `openai` client is `tests/gate/test_server.py`.
"""
import array
import http.client
import io
import json
import struct
import threading
import wave
from pathlib import Path

import pytest

import loom
from loom import server as srv
from loom._interfaces import Audio
from loom import Segment, Transcription


# -- stand-ins ----------------------------------------------------------------------------------------

class FakeStream:
    def __init__(self, ids):
        self._ids = ids
        self.tokens = []

    def __iter__(self):
        for i in self._ids:
            self.tokens.append(i)
            yield chr(i)


class FakeLM:
    """Characters are tokens; the reply is fixed, and generation stops at its end or the cap."""

    def __init__(self, reply="Hello there.", streams=True, roles=("user", "assistant", "system")):
        self.contract = {"interface": "text2text", "task": "text-generation"}
        self.path = Path("lm.gguf")
        self.reply, self.can_stream_tokens, self.chat_roles = reply, streams, list(roles)
        self.eos_token_ids = [0]
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return "".join(f"<{r}>{c}" for r, c in messages) + "<assistant>"

    def tokenize(self, text):
        return [ord(c) for c in text]

    def detokenize(self, ids):
        return "".join(chr(i) for i in ids)

    def generate_ids(self, ids, max_new_tokens, **sampling):
        self.calls.append(("ids", list(ids), max_new_tokens, sampling))
        return [ord(c) for c in self.reply][:max_new_tokens]

    def generate_stream(self, ids, max_new_tokens, **sampling):
        self.calls.append(("stream", list(ids), max_new_tokens, sampling))
        return FakeStream([ord(c) for c in self.reply][:max_new_tokens])


class FakeTTSDoor:
    def __init__(self, model):
        self.model = model

    def _declares(self, key):
        return key in self.model.declared

    def infer(self, text, **kwargs):
        self.model.calls.append(("infer", text, kwargs))
        return Audio(samples=[0.5, -0.5] * 100, sample_rate=24000)

    def stream(self, text, **kwargs):
        self.model.calls.append(("stream", text, kwargs))

        class S:
            sample_rate = 24000

            def __iter__(self):
                for _ in range(3):
                    yield Audio(samples=[0.25] * 50, sample_rate=24000)
        return S()


class FakeTTS:
    def __init__(self, voices=("alba", "bill"), driver="local s = inputs.speed or 1", declared=()):
        self.contract = {"interface": "text2speech", "task": "text-to-speech", "sample_rate": 24000}
        self.path = Path("tts.gguf")
        self.voices, self.driver_source, self.declared = list(voices), driver, set(declared)
        self.calls = []

    @property
    def text2speech(self):
        return FakeTTSDoor(self)


class FakeASR:
    def __init__(self, translates=False):
        self.contract = {"interface": "speech2text", "task": "automatic-speech-recognition", "sample_rate": 16000}
        self.path = Path("asr.gguf")
        self.translates = translates
        self.calls = []

    def transcribe(self, samples, language=None, task=None, target_language=None, timestamps=False):
        self.calls.append(dict(n=len(samples), language=language, task=task, target_language=target_language,
                               timestamps=timestamps))
        if task == "translate" and not self.translates:
            raise loom.LoomError("no task tokens")
        return Transcription(text=" Hello world. Again.", windows=1, timestamped=True,
                             segments=[Segment(0.0, 1.5, " Hello world.", True),
                                       Segment(1.5, 3725.25, " Again.", True)])


# -- a running server ---------------------------------------------------------------------------------

@pytest.fixture
def serve():
    servers = []

    def start(models, **options):
        s = srv.make_server(models, port=0, quiet=True, **options)
        threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        servers.append(s)
        return s.server_address[1]

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def request(port, method, path, body=None, headers=None, raw=False):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    conn.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
    resp = conn.getresponse()
    payload = resp.read()
    conn.close()
    if raw:
        return resp, payload
    return resp.status, json.loads(payload) if payload else None


def events(payload):
    out = []
    for block in payload.decode().split("\n\n"):
        if block.startswith("data: "):
            data = block[len("data: "):]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


def wav_bytes(samples, rate, channels=1, width=2):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"".join(int(s * 32767).to_bytes(2, "little", signed=True) for s in samples))
    return out.getvalue()


def multipart(fields, file_bytes, filename="clip.wav"):
    boundary = "loomtestboundary"
    parts = []
    for name, value in fields.items():
        for v in value if isinstance(value, list) else [value]:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                 f'Content-Type: audio/wav\r\n\r\n'.encode() + file_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


# -- which model answers ------------------------------------------------------------------------------

class TestRouting:
    def test_models_lists_each_with_its_door(self, serve):
        port = serve({"lm": FakeLM(), "tts": FakeTTS()})
        status, body = request(port, "GET", "/v1/models")
        assert status == 200 and body["object"] == "list"
        assert [(m["id"], m["interface"], m["object"]) for m in body["data"]] == [
            ("lm", "text2text", "model"), ("tts", "text2speech", "model")]
        assert request(port, "GET", "/v1/models/tts")[1]["task"] == "text-to-speech"
        assert request(port, "GET", "/v1/models/nope")[0] == 404
        assert request(port, "GET", "/health") == (200, {"status": "ok"})
        status, body = request(port, "POST", "/v1/embeddings", {})
        assert status == 404 and body["error"]["code"] == "not_found"

    def test_an_unknown_name_is_the_one_model_with_the_door(self, serve):
        port = serve({"lm": FakeLM(), "tts": FakeTTS()})
        status, body = request(port, "POST", "/v1/chat/completions",
                               {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
        assert status == 200 and body["model"] == "lm"

    def test_two_candidates_are_named_not_guessed(self, serve):
        port = serve({"a": FakeLM(), "b": FakeLM()})
        status, body = request(port, "POST", "/v1/chat/completions",
                               {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        assert status == 404 and body["error"]["code"] == "model_not_found"
        assert "'a', 'b'" in body["error"]["message"]
        assert request(port, "POST", "/v1/chat/completions",
                       {"model": "b", "messages": [{"role": "user", "content": "hi"}]})[1]["model"] == "b"

    def test_a_named_model_with_another_door_is_refused(self, serve):
        port = serve({"lm": FakeLM(), "tts": FakeTTS()})
        status, body = request(port, "POST", "/v1/audio/speech", {"model": "lm", "input": "x"})
        assert status == 400 and body["error"]["code"] == "model_not_supported"
        assert "/v1/chat/completions" in body["error"]["message"]

    def test_an_api_key_is_required_when_set(self, serve):
        port = serve({"lm": FakeLM()}, api_key="sekrit")
        status, body = request(port, "GET", "/v1/models")
        assert status == 401 and body["error"]["type"] == "authentication_error"
        assert request(port, "GET", "/v1/models", headers={"Authorization": "Bearer sekrit"})[0] == 200


# -- text ---------------------------------------------------------------------------------------------

def chat(port, **body):
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return request(port, "POST", "/v1/chat/completions", body)


class TestChat:
    def test_a_reply_in_openais_shape(self, serve):
        lm = FakeLM()
        port = serve({"lm": lm})
        status, body = chat(port, messages=[{"role": "developer", "content": "Be brief."},
                                            {"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                            temperature=0, seed=7, top_p=0.9)
        assert status == 200 and body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
        assert body["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "Hello there."},
                                    "logprobs": None, "finish_reason": "stop"}]
        prompt = "<system>Be brief.<user>hi<assistant>"
        assert body["usage"] == {"prompt_tokens": len(prompt), "completion_tokens": 12, "total_tokens": len(prompt) + 12}
        # The template is the file's; the sampling options are the request's, and nothing else.
        assert lm.calls == [("ids", [ord(c) for c in prompt], 512, {"temperature": 0.0, "top_p": 0.9, "seed": 7})]

    def test_the_cap_is_a_length_finish(self, serve):
        port = serve({"lm": FakeLM()}, max_tokens=5)
        body = chat(port)[1]
        assert body["choices"][0]["message"]["content"] == "Hello" and body["choices"][0]["finish_reason"] == "length"
        assert chat(port, max_completion_tokens=3)[1]["usage"]["completion_tokens"] == 3

    def test_a_stop_string_cuts_the_reply(self, serve):
        port = serve({"lm": FakeLM()})
        body = chat(port, stop=["ere", "zzz"])[1]
        assert body["choices"][0]["message"]["content"] == "Hello th"
        assert body["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.parametrize("bad, param", [
        ({"n": 2}, "n"),
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "tool", "content": "x"}]}, "messages"),
        ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}, "messages"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"stop": [1]}, "stop"),
        ({"temperature": "hot"}, "temperature"),
    ])
    def test_a_request_it_cannot_serve_is_a_400(self, serve, bad, param):
        port = serve({"lm": FakeLM()})
        status, body = chat(port, **bad)
        assert status == 400 and body["error"]["param"] == param

    def test_a_base_model_has_no_chat(self, serve):
        port = serve({"lm": FakeLM(roles=())})
        status, body = chat(port)
        assert status == 400 and "/v1/completions" in body["error"]["message"]

    def test_the_stream_is_the_reply_a_piece_at_a_time(self, serve):
        port = serve({"lm": FakeLM()})
        resp, payload = request(port, "POST", "/v1/chat/completions",
                                {"messages": [{"role": "user", "content": "hi"}], "stream": True,
                                 "stream_options": {"include_usage": True}}, raw=True)
        assert resp.status == 200 and resp.getheader("Content-Type") == "text/event-stream"
        got = events(payload)
        assert got[-1] == "[DONE]"
        chunks = got[:-1]
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        assert text == "Hello there."
        assert len([c for c in chunks if c["choices"] and c["choices"][0]["delta"].get("content")]) == 12
        assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "stop"
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 12

    def test_a_stop_string_split_across_pieces_is_held_back(self, serve):
        port = serve({"lm": FakeLM()})
        _, payload = request(port, "POST", "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"}], "stream": True, "stop": "there"},
                             raw=True)
        chunks = events(payload)[:-1]
        assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Hello "
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_a_file_that_cannot_stream_answers_in_one_piece(self, serve):
        lm = FakeLM(streams=False)
        port = serve({"lm": lm})
        _, payload = request(port, "POST", "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"}], "stream": True}, raw=True)
        contents = [c["choices"][0]["delta"].get("content") for c in events(payload)[:-1]]
        assert contents == ["", "Hello there.", None]
        assert lm.calls[0][0] == "ids"


class TestCompletions:
    def test_a_prompt_is_continued_as_it_is(self, serve):
        lm = FakeLM(reply="0123456789abcdefghij")
        port = serve({"lm": lm})
        body = request(port, "POST", "/v1/completions", {"prompt": "ab"})[1]
        # OpenAI's default for this endpoint is 16 tokens.
        assert body["object"] == "text_completion" and body["choices"][0]["text"] == "0123456789abcdef"
        assert body["choices"][0]["finish_reason"] == "length"
        assert lm.calls[-1][1] == [97, 98]
        request(port, "POST", "/v1/completions", {"prompt": [5, 6, 7], "max_tokens": 2})
        assert lm.calls[-1][1:3] == ([5, 6, 7], 2)

    def test_streamed(self, serve):
        port = serve({"lm": FakeLM()})
        _, payload = request(port, "POST", "/v1/completions", {"prompt": "x", "stream": True, "max_tokens": 5},
                             raw=True)
        chunks = events(payload)[:-1]
        assert "".join(c["choices"][0]["text"] for c in chunks) == "Hello"
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"


# -- speech -------------------------------------------------------------------------------------------

class TestSpeech:
    def test_wav_at_the_models_rate(self, serve):
        tts = FakeTTS()
        port = serve({"tts": tts})
        resp, payload = request(port, "POST", "/v1/audio/speech",
                                {"model": "tts-1", "input": "Hi.", "voice": "bill", "response_format": "wav",
                                 "speed": 1.5}, raw=True)
        assert resp.status == 200 and resp.getheader("Content-Type") == "audio/wav"
        with wave.open(io.BytesIO(payload)) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (24000, 1, 2, 200)
            first = struct.unpack("<h", w.readframes(1))[0]
        assert first == int(0.5 * 32767)
        assert tts.calls == [("infer", "Hi.", {"voice": "bill", "speed": 1.5})]

    def test_pcm_streams_a_piece_at_a_time(self, serve):
        tts = FakeTTS()
        port = serve({"tts": tts})
        resp, payload = request(port, "POST", "/v1/audio/speech",
                                {"input": "Hi. There.", "response_format": "pcm"}, raw=True)
        assert resp.getheader("Transfer-Encoding") == "chunked" and resp.getheader("X-Sample-Rate") == "24000"
        assert len(payload) == 3 * 50 * 2 and tts.calls[0][0] == "stream"

    def test_without_ffmpeg_the_default_is_wav_and_mp3_is_refused(self, serve, monkeypatch):
        monkeypatch.setattr(srv, "_ffmpeg", lambda: None)
        port = serve({"tts": FakeTTS()})
        resp, _ = request(port, "POST", "/v1/audio/speech", {"input": "Hi."}, raw=True)
        assert resp.getheader("Content-Type") == "audio/wav"
        status, body = request(port, "POST", "/v1/audio/speech", {"input": "Hi.", "response_format": "mp3"})
        assert status == 400 and "ffmpeg" in body["error"]["message"]

    def test_a_voice_the_model_lacks_lists_the_ones_it_has(self, serve):
        port = serve({"tts": FakeTTS()})
        status, body = request(port, "POST", "/v1/audio/speech", {"input": "Hi.", "voice": "alloy"})
        assert status == 400 and body["error"]["param"] == "voice" and "alba, bill" in body["error"]["message"]

    def test_a_voice_is_ignored_by_a_model_without_voices(self, serve):
        tts = FakeTTS(voices=())
        port = serve({"tts": tts})
        request(port, "POST", "/v1/audio/speech", {"input": "Hi.", "voice": "alloy", "response_format": "wav"},
                raw=True)
        assert tts.calls[0][2] == {}

    def test_speed_and_instructions_only_where_the_file_takes_them(self, serve):
        port = serve({"tts": FakeTTS(driver="return inputs.tokens")})
        status, body = request(port, "POST", "/v1/audio/speech", {"input": "Hi.", "speed": 2})
        assert status == 400 and body["error"]["param"] == "speed"
        status, body = request(port, "POST", "/v1/audio/speech", {"input": "Hi.", "instructions": "whisper"})
        assert status == 400 and body["error"]["param"] == "instructions"
        tts = FakeTTS(declared={"tts.instruct"})
        port = serve({"tts": tts})
        request(port, "POST", "/v1/audio/speech",
                {"input": "Hi.", "instructions": "low pitch", "response_format": "wav"}, raw=True)
        assert tts.calls[0][2] == {"instruct": "low pitch"}

    def test_the_driver_input_check(self):
        model = type("M", (), {})()
        for source, reads in (("x = inputs.speed", True), ("x = inputs['speed']", True),
                              ('x = inputs [ "speed" ]', True), ("x = inputs.speedup", False),
                              ("speed = 1", False)):
            model.driver_source = source
            assert srv._driver_reads(model, "speed") is reads, source


# -- transcription ------------------------------------------------------------------------------------

class TestTranscriptions:
    def post(self, port, fields, data, path="/v1/audio/transcriptions"):
        body, headers = multipart(fields, data)
        return request(port, "POST", path, body, headers, raw=True)

    def test_json_text_and_the_samples_it_heard(self, serve):
        asr = FakeASR()
        port = serve({"asr": asr})
        resp, payload = self.post(port, {"model": "whisper-1", "language": "en"}, wav_bytes([0.1] * 1600, 16000))
        assert resp.status == 200 and json.loads(payload) == {"text": " Hello world. Again."}
        assert asr.calls == [dict(n=1600, language="en", task=None, target_language=None, timestamps=False)]
        resp, payload = self.post(port, {"response_format": "text"}, wav_bytes([0.1] * 16, 16000))
        assert payload.decode() == " Hello world. Again."

    def test_subtitles_and_verbose_json(self, serve):
        asr = FakeASR()
        port = serve({"asr": asr})
        _, srt = self.post(port, {"response_format": "srt"}, wav_bytes([0.0] * 32000, 16000))
        assert srt.decode() == ("1\n00:00:00,000 --> 00:00:01,500\nHello world.\n\n"
                                "2\n00:00:01,500 --> 01:02:05,250\nAgain.\n")
        _, vtt = self.post(port, {"response_format": "vtt"}, wav_bytes([0.0] * 32000, 16000))
        assert vtt.decode().startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello world.\n")
        _, verbose = self.post(port, {"response_format": "verbose_json", "timestamp_granularities[]": ["segment"]},
                               wav_bytes([0.0] * 32000, 16000))
        body = json.loads(verbose)
        assert body["duration"] == 2.0 and body["task"] == "transcribe"
        assert [(s["id"], s["start"], s["end"]) for s in body["segments"]] == [(0, 0.0, 1.5), (1, 1.5, 3725.25)]
        assert asr.calls[-1]["timestamps"] is True

    def test_another_rate_needs_ffmpeg(self, serve, monkeypatch):
        monkeypatch.setattr(srv, "_ffmpeg", lambda: None)
        port = serve({"asr": FakeASR()})
        resp, payload = self.post(port, {}, wav_bytes([0.0] * 100, 44100))
        assert resp.status == 400 and "44100 Hz WAV" in json.loads(payload)["error"]["message"]
        resp, payload = self.post(port, {}, b"ID3 not a wav")
        assert resp.status == 400 and "ffmpeg" in json.loads(payload)["error"]["message"]

    def test_translation_falls_back_to_a_target_language(self, serve):
        asr = FakeASR()
        port = serve({"asr": asr})
        resp, payload = self.post(port, {}, wav_bytes([0.0] * 10, 16000), path="/v1/audio/translations")
        assert resp.status == 200
        assert [(c["task"], c["target_language"]) for c in asr.calls] == [("translate", None), (None, "en")]
        asr = FakeASR(translates=True)
        port = serve({"asr": asr})
        self.post(port, {}, wav_bytes([0.0] * 10, 16000), path="/v1/audio/translations")
        assert [(c["task"], c["target_language"]) for c in asr.calls] == [("translate", None)]

    def test_a_body_that_is_not_multipart(self, serve):
        port = serve({"asr": FakeASR()})
        status, body = request(port, "POST", "/v1/audio/transcriptions", {"file": "x"})
        assert status == 400 and "multipart" in body["error"]["message"]


# -- the pieces ---------------------------------------------------------------------------------------

class TestPieces:
    def test_read_wav_formats(self):
        stereo = wav_bytes([0.5, -0.5, 0.25, 0.25], 8000, channels=2)
        samples, rate = srv.read_wav(stereo)
        assert rate == 8000 and samples == pytest.approx([0.0, 0.25], abs=1e-4)

        def riff(code, bits, payload, rate=16000):
            fmt = struct.pack("<HHIIHH", code, 1, rate, rate * bits // 8, bits // 8, bits)
            return (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(payload)) + b"WAVE"
                    + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload)

        f32 = array.array("f", [0.5, -1.0]).tobytes()
        assert srv.read_wav(riff(3, 32, f32))[0] == [0.5, -1.0]
        i24 = (4194304).to_bytes(3, "little", signed=True) + (-8388608).to_bytes(3, "little", signed=True)
        assert srv.read_wav(riff(1, 24, i24))[0] == [0.5, -1.0]
        assert srv.read_wav(riff(1, 8, bytes([192, 0])))[0] == [0.5, -1.0]
        assert srv.read_wav(b"OggS....") is None

    def test_stop_scan(self):
        scan = srv._StopScan(["END"])
        assert [scan.feed(p) for p in ("ab", "cE", "N", "Dzz")] == ["", "ab", "c", ""]
        assert scan.stopped and scan.feed("more") == ""
        scan = srv._StopScan([])
        assert scan.feed("abc") == "abc" and scan.flush() == ""

    def test_the_command_names_each_model(self, tmp_path, monkeypatch):
        loaded = []
        monkeypatch.setattr(loom.Model, "from_file", classmethod(lambda cls, p, device="": loaded.append(p) or p))
        monkeypatch.setattr(loom.Model, "from_pretrained",
                            classmethod(lambda cls, repo, filename=None, device="": (repo, filename)))
        f = tmp_path / "smollm2-360m.gguf"
        f.write_bytes(b"")
        assert srv._load(str(f), "") == ("smollm2-360m", f)
        assert srv._load(f"chat={f}", "")[0] == "chat"
        assert srv._load("org/kokoro-82m-loom", "") == ("kokoro-82m-loom", ("org/kokoro-82m-loom", None))
        assert srv._load("tts=org/repo:a.gguf", "") == ("tts", ("org/repo", "a.gguf"))
        with pytest.raises(SystemExit):
            srv._load("no-such-file.gguf", "")
