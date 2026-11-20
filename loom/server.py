"""An OpenAI-compatible HTTP server over loom models (loom-py#52).

    loom-serve smollm2=smollm2-360m-instruct.gguf whisper=moonshine-tiny-q8_0.gguf paradee-8m-q8_0_pc.gguf

Then any OpenAI client talks to it:

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
    client.chat.completions.create(model="smollm2", messages=[{"role": "user", "content": "Hi"}])

Each endpoint is one of the doors a model already declares (loom.cpp ADR-013), and the server picks
nothing by architecture:

==============================  ============  ===================================================
endpoint                        door          what it calls
==============================  ============  ===================================================
``POST /v1/chat/completions``   text2text     the file's chat template, then ``generate_ids`` /
                                              ``generate_stream``
``POST /v1/completions``        text2text     the prompt as it is
``POST /v1/audio/speech``       text2speech   ``text2speech.infer`` / ``.stream`` (``pcm`` streams)
``POST /v1/audio/transcriptions`` speech2text ``Model.transcribe``
``POST /v1/audio/translations`` speech2text   ``Model.transcribe`` into English
``GET /v1/models``              --            what is loaded
==============================  ============  ===================================================

**No dependencies**, as the rest of this package: the standard library's ``http.server``, and ffmpeg
from ``PATH`` when a request needs an audio format other than WAV and raw PCM (mp3, opus, aac, flac out;
anything ffmpeg reads in, or a WAV at another sample rate than the model's). Without ffmpeg those
requests fail with a 400 that says so, and ``/v1/audio/speech`` with no ``response_format`` answers WAV
instead of OpenAI's default mp3.

**Which model answers.** The request's ``model`` names one of the served models. When it names none
that is loaded (an OpenAI client's ``"whisper-1"``) and exactly one served model has the endpoint's door,
that one answers and the response says its real name; with several, the request is refused with the
names. The same rule as a ``device`` argument: the caller's name, else the one unambiguous answer.

**One request per model at a time.** A model holds one set of caches, so its requests queue on a lock;
two served models run concurrently.

Bound to 127.0.0.1 by default. ``--api-key`` (or ``$LOOM_API_KEY``) makes every request carry
``Authorization: Bearer <key>``.
"""
from __future__ import annotations

import argparse
import array
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from email.parser import BytesParser
from email.policy import HTTP as _HTTP_POLICY
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

#: The endpoint each door answers, for error messages and `/v1/models`.
DOORS = {"text2text": "/v1/chat/completions, /v1/completions",
         "text2speech": "/v1/audio/speech",
         "speech2text": "/v1/audio/transcriptions, /v1/audio/translations"}

#: `response_format` of `/v1/audio/speech` -> (Content-Type, ffmpeg output arguments or None).
SPEECH_FORMATS = {
    "wav": ("audio/wav", None),
    "pcm": ("audio/pcm", None),
    "mp3": ("audio/mpeg", ["-c:a", "libmp3lame", "-q:a", "2", "-f", "mp3"]),
    "opus": ("audio/ogg", ["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"]),
    "aac": ("audio/aac", ["-c:a", "aac", "-b:a", "128k", "-f", "adts"]),
    "flac": ("audio/flac", ["-c:a", "flac", "-f", "flac"]),
}

TRANSCRIPTION_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")


class ApiError(Exception):
    """A request the server refuses, as OpenAI's error object: ``{"error": {"message", "type", ...}}``."""

    def __init__(self, status: int, message: str, *, type: str = "invalid_request_error",
                 param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.status, self.message, self.type, self.param, self.code = status, message, type, param, code

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


class Served:
    """One loaded model, under the name requests use for it."""

    def __init__(self, name: str, model: Any):
        self.name = name
        self.model = model
        self.interface = model.contract.get("interface") or ""
        self.lock = threading.Lock()
        try:
            self.created = int(Path(model.path).stat().st_mtime)
        except (OSError, TypeError, AttributeError):
            self.created = int(time.time())

    def card(self) -> dict:
        return {"id": self.name, "object": "model", "created": self.created, "owned_by": "loom",
                "task": self.model.contract.get("task") or "", "interface": self.interface}


class Registry:
    """The served models, and the rule for which one answers a request."""

    def __init__(self, models: Mapping[str, Any]):
        if not models:
            raise ValueError("serve at least one model")
        self.served = {name: Served(name, m) for name, m in models.items()}

    def pick(self, name: str | None, interface: str) -> Served:
        found = self.served.get(name) if name else None
        if found is not None:
            if found.interface != interface:
                raise ApiError(400, f"model {name!r} is {found.interface or 'undeclared'}, and this endpoint "
                               f"needs a {interface} model. Its door is {DOORS.get(found.interface, 'none here')}.",
                               param="model", code="model_not_supported")
            return found
        candidates = [s for s in self.served.values() if s.interface == interface]
        if len(candidates) == 1:
            return candidates[0]
        names = ", ".join(repr(s.name) for s in candidates) or "none"
        if name:
            message = f"model {name!r} is not served. Served {interface} models: {names}."
        else:
            message = f"name a model: served {interface} models: {names}."
        raise ApiError(404, message, param="model", code="model_not_found")


# -- text ---------------------------------------------------------------------------------------------

def _messages(raw: Any) -> list[tuple[str, str]]:
    """OpenAI's messages as the (role, text) pairs a chat template takes."""
    if not isinstance(raw, list) or not raw:
        raise ApiError(400, "`messages` is a non-empty list", param="messages")
    out = []
    for i, m in enumerate(raw):
        if not isinstance(m, Mapping) or "role" not in m:
            raise ApiError(400, f"messages[{i}] has no role", param="messages")
        role = str(m["role"])
        # `developer` is what OpenAI's newer models call the system turn.
        role = "system" if role == "developer" else role
        if role not in ("system", "user", "assistant"):
            raise ApiError(400, f"messages[{i}] has role {role!r}; a chat template here takes system, "
                           f"user and assistant turns (no tools).", param="messages")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, Mapping) or part.get("type") != "text":
                    kind = part.get("type") if isinstance(part, Mapping) else type(part).__name__
                    raise ApiError(400, f"messages[{i}] has a {kind!r} part; only text is read.",
                                   param="messages")
                parts.append(str(part.get("text", "")))
            content = "".join(parts)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            raise ApiError(400, f"messages[{i}].content is neither text nor a list of parts", param="messages")
        out.append((role, content))
    return out


def _stops(raw: Any) -> list[str]:
    if raw is None:
        return []
    stops = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, list) else None
    if stops is None or not all(isinstance(s, str) for s in stops):
        raise ApiError(400, "`stop` is a string or a list of strings", param="stop")
    return [s for s in stops if s]


def _sampling(req: Mapping) -> dict:
    """The sampling options a request names. Unset is the checkpoint's own generation config, as for
    `Model.generate`; `temperature: 0` is greedy, as every driver reads it."""
    out = {}
    for key, cast in (("temperature", float), ("top_p", float), ("top_k", int), ("seed", int)):
        value = req.get(key)
        if value is not None:
            try:
                out[key] = cast(value)
            except (TypeError, ValueError):
                raise ApiError(400, f"`{key}` is a number", param=key) from None
    return out


def _max_tokens(req: Mapping, default: int) -> int:
    value = req.get("max_completion_tokens", req.get("max_tokens"))
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ApiError(400, "`max_tokens` is a positive integer", param="max_tokens")
    return value


class _StopScan:
    """Cuts generated text at the first stop string, holding back what could still begin one."""

    def __init__(self, stops: Sequence[str]):
        self.stops = list(stops)
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.pending = ""
        self.stopped = False

    def feed(self, piece: str) -> str:
        """The text that is safe to hand back after `piece`."""
        if self.stopped:
            return ""
        self.pending += piece
        cut = min((i for i in (self.pending.find(s) for s in self.stops) if i >= 0), default=-1)
        if cut >= 0:
            out, self.pending, self.stopped = self.pending[:cut], "", True
            return out
        if not self.hold:
            out, self.pending = self.pending, ""
            return out
        out, self.pending = self.pending[:-self.hold], self.pending[-self.hold:]
        return out

    def flush(self) -> str:
        out, self.pending = self.pending, ""
        return out


class TextJob:
    """One text generation, run whole or streamed, with what the response reports about it."""

    def __init__(self, model: Any, prompt_ids: list[int], max_tokens: int, stops: list[str], sampling: dict):
        self.model, self.prompt_ids, self.max_tokens = model, prompt_ids, max_tokens
        self.stops, self.sampling = stops, sampling
        self.completion_tokens = 0
        self.finish_reason = "stop"

    def _finish(self, ids: Sequence[int], stopped: bool) -> None:
        self.completion_tokens = len(ids)
        # The returned ids end where the driver stopped: below the ceiling, it drew a stop token.
        self.finish_reason = "stop" if stopped or len(ids) < self.max_tokens else "length"

    def _ids(self, ids: Sequence[int]) -> list[int]:
        stops = set(self.model.eos_token_ids)
        ids = list(ids)
        while ids and ids[-1] in stops:
            ids.pop()
        return ids

    def run(self) -> str:
        ids = self._ids(self.model.generate_ids(self.prompt_ids, max_new_tokens=self.max_tokens, **self.sampling))
        scan = _StopScan(self.stops)
        text = scan.feed(self.model.detokenize(ids)) + ("" if scan.stopped else scan.flush())
        self._finish(ids, scan.stopped)
        return text

    def pieces(self) -> Iterator[str]:
        """The text as it is drawn: a token at a time where the file streams (`can_stream_tokens`),
        else the whole generation as one piece."""
        if not self.model.can_stream_tokens:
            text = self.run()
            if text:
                yield text
            return
        stream = self.model.generate_stream(self.prompt_ids, max_new_tokens=self.max_tokens, **self.sampling)
        scan = _StopScan(self.stops)
        for piece in stream:
            out = scan.feed(piece)
            if out:
                yield out
            if scan.stopped:
                break
        if not scan.stopped:
            tail = scan.flush()
            if tail:
                yield tail
        self._finish(self._ids(stream.tokens), scan.stopped)

    def usage(self) -> dict:
        p, c = len(self.prompt_ids), self.completion_tokens
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


# -- audio --------------------------------------------------------------------------------------------

def _ffmpeg() -> str | None:
    return shutil.which(os.environ.get("LOOM_FFMPEG", "ffmpeg"))


def _pcm16(samples: Iterable[float]) -> bytes:
    data = array.array("h", (int(max(-1.0, min(1.0, s)) * 32767.0) for s in samples))
    if sys.byteorder == "big":
        data.byteswap()
    return data.tobytes()


def _wav(pcm: bytes, rate: int, channels: int) -> bytes:
    out = io.BytesIO()
    import wave
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()


def _encode(pcm: bytes, rate: int, channels: int, fmt: str) -> bytes:
    if fmt == "pcm":
        return pcm
    if fmt == "wav":
        return _wav(pcm, rate, channels)
    exe = _ffmpeg()
    if exe is None:
        raise ApiError(400, f"response_format {fmt!r} needs ffmpeg on PATH, which this server does not have; "
                       f"ask for 'wav' or 'pcm'.", param="response_format")
    run = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(rate),
                          "-ac", str(channels), "-i", "pipe:0", *SPEECH_FORMATS[fmt][1], "pipe:1"],
                         input=pcm, capture_output=True)
    if run.returncode != 0:
        raise ApiError(500, f"ffmpeg could not encode {fmt}: {run.stderr.decode(errors='replace').strip()}",
                       type="server_error")
    return run.stdout


def read_wav(data: bytes) -> tuple[list[float], int] | None:
    """Mono floats and the rate of a RIFF WAV (integer PCM of 8-32 bits, or float), or None for anything
    else. Channels are averaged."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos, fmt, payload = 12, None, None
    while pos + 8 <= len(data):
        tag, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        body = data[pos + 8:pos + 8 + size]
        if tag == b"fmt ":
            fmt = body
        elif tag == b"data":
            payload = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or payload is None or len(fmt) < 16:
        return None
    code = int.from_bytes(fmt[0:2], "little")
    channels = int.from_bytes(fmt[2:4], "little")
    rate = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")
    if code == 0xFFFE and len(fmt) >= 26:  # WAVE_FORMAT_EXTENSIBLE: the real code opens the sub-format GUID
        code = int.from_bytes(fmt[24:26], "little")
    if channels < 1 or rate < 1:
        return None
    if code == 1 and bits in (8, 16, 24, 32):
        width = bits // 8
        n = len(payload) // width
        if bits == 8:
            values = [(b - 128) / 128.0 for b in payload[:n]]
        elif bits == 24:
            values = [int.from_bytes(payload[i:i + 3], "little", signed=True) / 8388608.0
                      for i in range(0, n * 3, 3)]
        else:
            a = array.array("h" if bits == 16 else "i", payload[:n * width])
            if sys.byteorder == "big":
                a.byteswap()
            scale = 32768.0 if bits == 16 else 2147483648.0
            values = [v / scale for v in a]
    elif code == 3 and bits in (32, 64):
        a = array.array("f" if bits == 32 else "d", payload[:len(payload) // (bits // 8) * (bits // 8)])
        if sys.byteorder == "big":
            a.byteswap()
        values = list(a)
    else:
        return None
    if channels > 1:
        frames = len(values) // channels
        values = [sum(values[f * channels:(f + 1) * channels]) / channels for f in range(frames)]
    return values, rate


def decode_audio(data: bytes, filename: str, rate: int) -> list[float]:
    """An uploaded file as mono floats at `rate`: a WAV at that rate here, anything else through ffmpeg."""
    wav = read_wav(data)
    if wav is not None and wav[1] == rate:
        return wav[0]
    exe = _ffmpeg()
    if exe is None:
        got = f"a {wav[1]} Hz WAV" if wav is not None else f"{filename or 'the file'}, which is not a WAV this reads"
        raise ApiError(400, f"this model hears {rate} Hz mono and got {got}; without ffmpeg on PATH the server "
                       f"reads only WAV at the model's rate.", param="file")
    suffix = Path(filename or "audio").suffix or ".bin"
    # A file rather than a pipe: mp4/m4a put their index at the end, and ffmpeg cannot seek a pipe. Closed
    # before ffmpeg opens it, which Windows needs.
    fd, name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        run = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-i", name, "-f", "f32le",
                              "-ac", "1", "-ar", str(rate), "pipe:1"], capture_output=True)
    finally:
        os.unlink(name)
    if run.returncode != 0:
        raise ApiError(400, f"ffmpeg could not read {filename or 'the file'}: "
                       f"{run.stderr.decode(errors='replace').strip()}", param="file")
    samples = array.array("f")
    samples.frombytes(run.stdout[:len(run.stdout) // 4 * 4])
    if sys.byteorder == "big":
        samples.byteswap()
    return list(samples)


def _timestamp(seconds: float, sep: str) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _subtitles(segments: Sequence[Any], vtt: bool) -> str:
    lines = ["WEBVTT", ""] if vtt else []
    for i, seg in enumerate(segments, 1):
        if not vtt:
            lines.append(str(i))
        sep = "." if vtt else ","
        lines.append(f"{_timestamp(seg.start, sep)} --> {_timestamp(seg.end, sep)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def _multipart(content_type: str, body: bytes) -> dict[str, Any]:
    """A multipart/form-data body as {name: str, or (filename, bytes) for a file}; a repeated name
    (`timestamp_granularities[]`) as a list."""
    message = BytesParser(policy=_HTTP_POLICY).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("latin-1") + body)
    if not message.is_multipart():
        raise ApiError(400, "expected a multipart/form-data body")
    fields: dict[str, Any] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        value: Any = (filename, payload) if filename is not None else payload.decode("utf-8", errors="replace")
        if name in fields:
            prior = fields[name]
            fields[name] = (prior if isinstance(prior, list) else [prior]) + [value]
        else:
            fields[name] = value
    return fields


# -- the server ---------------------------------------------------------------------------------------

class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, registry: Registry, *, api_key: str | None = None,
                 max_tokens: int = 512, quiet: bool = False):
        self.registry, self.api_key, self.max_tokens, self.quiet = registry, api_key, max_tokens, quiet
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: Server

    # -- plumbing ---------------------------------------------------------------------------------

    def log_message(self, format, *args):  # noqa: A002 -- the base class's name
        if not self.server.quiet:
            super().log_message(format, *args)

    def _send(self, status: int, body: bytes, content_type: str, headers: Mapping[str, str] = {}) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json")

    def _open_chunked(self, content_type: str, headers: Mapping[str, str] = {}) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()

    def _chunk(self, data: bytes) -> None:
        if data:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    def _close_chunked(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _event(self, obj: Any) -> None:
        self._chunk(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _json_body(self) -> dict:
        try:
            req = json.loads(self._body() or b"{}")
        except json.JSONDecodeError as e:
            raise ApiError(400, f"the body is not JSON: {e}") from None
        if not isinstance(req, dict):
            raise ApiError(400, "the body is a JSON object")
        return req

    def _authorise(self) -> None:
        key = self.server.api_key
        if key and self.headers.get("Authorization", "") != f"Bearer {key}":
            raise ApiError(401, "a valid API key is required: `Authorization: Bearer <key>`",
                           type="authentication_error", code="invalid_api_key")

    def _dispatch(self, routes: Mapping[str, Any]) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        streaming = False
        try:
            self._authorise()
            handler = routes.get(path)
            if handler is None and path.startswith("/v1/models/"):
                handler = self._model_card
            if handler is None:
                raise ApiError(404, f"no route {self.command} {path}", code="not_found")
            streaming = bool(handler())
        except ApiError as e:
            # An error may come before the body was read; the connection does not carry another request.
            self.close_connection = True
            if not streaming:
                self._json(e.status, e.body())
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client went away
        except Exception as e:  # noqa: BLE001 -- a request must get an answer, not a dropped socket
            from . import LoomError, UnsupportedTask
            known = (LoomError, UnsupportedTask, ValueError, TypeError, KeyError)
            error = ApiError(400, str(e)) if isinstance(e, known) else ApiError(
                500, f"{type(e).__name__}: {e}", type="server_error")
            self.close_connection = True
            self._json(error.status, error.body())

    def do_GET(self):  # noqa: N802
        self._dispatch({"/v1/models": self._models, "/health": self._health})

    def do_POST(self):  # noqa: N802
        self._dispatch({"/v1/chat/completions": self._chat, "/v1/completions": self._completions,
                        "/v1/audio/speech": self._speech, "/v1/audio/transcriptions": self._transcriptions,
                        "/v1/audio/translations": self._translations})

    # -- routes: each returns True when it answered by streaming ---------------------------------------

    def _health(self):
        self._json(200, {"status": "ok"})

    def _models(self):
        self._json(200, {"object": "list", "data": [s.card() for s in self.server.registry.served.values()]})

    def _model_card(self):
        name = self.path.split("?", 1)[0].rstrip("/")[len("/v1/models/"):]
        served = self.server.registry.served.get(name)
        if served is None:
            raise ApiError(404, f"model {name!r} is not served", param="model", code="model_not_found")
        self._json(200, served.card())

    def _text(self, chat: bool):
        req = self._json_body()
        served = self.server.registry.pick(req.get("model"), "text2text")
        model = served.model
        if req.get("n", 1) != 1:
            raise ApiError(400, "`n` must be 1", param="n")
        if chat:
            if not model.chat_roles:
                raise ApiError(400, f"model {served.name!r} has no chat template (a base model); "
                               f"/v1/completions continues a prompt.", param="model")
            prompt = model.apply_chat_template(_messages(req.get("messages")), add_generation_prompt=True)
            prompt_ids = list(model.tokenize(prompt))
        else:
            prompt = req.get("prompt")
            if isinstance(prompt, list) and len(prompt) == 1:
                prompt = prompt[0]
            if isinstance(prompt, str):
                prompt_ids = list(model.tokenize(prompt))
            elif isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
                prompt_ids = list(prompt)
            else:
                raise ApiError(400, "`prompt` is one string or one list of token ids", param="prompt")
        # OpenAI's completions default is 16 tokens; chat's is the context, which here is the server's cap.
        default = self.server.max_tokens if chat else 16
        job = TextJob(model, prompt_ids, _max_tokens(req, default), _stops(req.get("stop")), _sampling(req))
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())
        head = {"id": cid, "created": created, "model": served.name}
        if not req.get("stream"):
            with served.lock:
                text = job.run()
            if chat:
                choice = {"index": 0, "message": {"role": "assistant", "content": text},
                          "logprobs": None, "finish_reason": job.finish_reason}
            else:
                choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": job.finish_reason}
            self._json(200, {**head, "object": "chat.completion" if chat else "text_completion",
                             "choices": [choice], "usage": job.usage()})
            return False
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        kind = "chat.completion.chunk" if chat else "text_completion"

        def chunk(delta: str | None, finish: str | None, role: bool = False) -> dict:
            if chat:
                d = {"role": "assistant", "content": ""} if role else ({} if delta is None else {"content": delta})
                choice = {"index": 0, "delta": d, "logprobs": None, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": delta or "", "logprobs": None, "finish_reason": finish}
            out = {**head, "object": kind, "choices": [choice]}
            if include_usage:
                out["usage"] = None
            return out

        with served.lock:
            pieces = job.pieces()
            first = next(pieces, None)  # an error before any text is still an ordinary error response
            self._open_chunked("text/event-stream")
            try:
                if chat:
                    self._event(chunk(None, None, role=True))
                if first is not None:
                    self._event(chunk(first, None))
                for piece in pieces:
                    self._event(chunk(piece, None))
                self._event(chunk(None, job.finish_reason))
                if include_usage:
                    self._event({**head, "object": kind, "choices": [], "usage": job.usage()})
            except (BrokenPipeError, ConnectionResetError):
                return True
            except Exception as e:  # noqa: BLE001 -- headers are out: the error travels as an event
                self._event(ApiError(500, f"{type(e).__name__}: {e}", type="server_error").body())
            self._chunk(b"data: [DONE]\n\n")
            self._close_chunked()
        return True

    def _chat(self):
        return self._text(chat=True)

    def _completions(self):
        return self._text(chat=False)

    def _speech(self):
        req = self._json_body()
        served = self.server.registry.pick(req.get("model"), "text2speech")
        model = served.model
        text = req.get("input")
        if not isinstance(text, str) or not text.strip():
            raise ApiError(400, "`input` is the text to speak", param="input")
        fmt = req.get("response_format") or ("mp3" if _ffmpeg() else "wav")
        if fmt not in SPEECH_FORMATS:
            raise ApiError(400, f"response_format {fmt!r} is not one of {', '.join(SPEECH_FORMATS)}",
                           param="response_format")
        kwargs = self._speech_inputs(served, req)
        door = model.text2speech
        content_type = SPEECH_FORMATS[fmt][0]
        with served.lock:
            if fmt == "pcm":
                # The one format with no header to write first: it streams a piece at a time.
                stream = door.stream(text, **kwargs)
                rate = stream.sample_rate
                pieces = iter(stream)
                first = next(pieces, None)
                self._open_chunked(content_type, {"X-Sample-Rate": str(rate)})
                try:
                    if first is not None:
                        self._chunk(_pcm16(first.samples))
                    for piece in pieces:
                        self._chunk(_pcm16(piece.samples))
                except (BrokenPipeError, ConnectionResetError):
                    return True
                except Exception:  # noqa: BLE001 -- headers are out; a short body is the only signal left
                    self.close_connection = True
                    return True
                self._close_chunked()
                return True
            audio = door.infer(text, **kwargs)
        body = _encode(_pcm16(audio.samples), audio.sample_rate, audio.channels, fmt)
        self._send(200, body, content_type, {"X-Sample-Rate": str(audio.sample_rate)})
        return False

    def _speech_inputs(self, served: Served, req: Mapping) -> dict:
        model = served.model
        kwargs: dict[str, Any] = {}
        voice = req.get("voice")
        if isinstance(voice, Mapping):  # OpenAI's custom-voice form, {"id": ...}
            voice = voice.get("id")
        voices = list(model.voices)
        if voice and voices:
            if voice not in voices:
                raise ApiError(400, f"model {served.name!r} has no voice {voice!r}; its voices: "
                               f"{', '.join(voices)}", param="voice")
            kwargs["voice"] = voice
        speed = req.get("speed")
        if speed is not None and float(speed) != 1.0:
            if not _driver_reads(model, "speed"):
                raise ApiError(400, f"model {served.name!r} has no speed control", param="speed")
            if not 0.25 <= float(speed) <= 4.0:
                raise ApiError(400, "`speed` is between 0.25 and 4.0", param="speed")
            kwargs["speed"] = float(speed)
        instructions = req.get("instructions")
        if instructions:
            if not model.text2speech._declares("tts.instruct"):
                raise ApiError(400, f"model {served.name!r} takes no voice instructions", param="instructions")
            kwargs["instruct"] = str(instructions)
        language = req.get("language")  # not OpenAI's; a multilingual file's language code
        if language:
            kwargs["language"] = str(language)
        return kwargs

    def _transcriptions(self, translate: bool = False):
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            raise ApiError(400, "send the audio as multipart/form-data, with the file in `file`")
        fields = _multipart(content_type, self._body())
        upload = fields.get("file")
        if not isinstance(upload, tuple):
            raise ApiError(400, "`file` is the audio to transcribe", param="file")
        served = self.server.registry.pick(fields.get("model") or None, "speech2text")
        model = served.model
        fmt = fields.get("response_format") or "json"
        if fmt not in TRANSCRIPTION_FORMATS:
            raise ApiError(400, f"response_format {fmt!r} is not one of {', '.join(TRANSCRIPTION_FORMATS)}",
                           param="response_format")
        rate = int(model.contract.get("sample_rate") or 16000)
        samples = decode_audio(upload[1], upload[0], rate)
        language = fields.get("language") or None
        timestamps = fmt in ("verbose_json", "srt", "vtt")
        with served.lock, warnings.catch_warnings():
            # `language` on a monolingual file is ignored with a warning (Model.transcribe); a server
            # has no one to warn, and the transcript is what was asked for.
            warnings.simplefilter("ignore", RuntimeWarning)
            if translate:
                result = self._translate(model, samples, language, timestamps)
            else:
                result = model.transcribe(samples, language=language, timestamps=timestamps)
        if fmt == "json":
            self._json(200, {"text": result.text})
        elif fmt == "text":
            self._send(200, result.text.encode("utf-8"), "text/plain; charset=utf-8")
        elif fmt == "verbose_json":
            self._json(200, {
                "task": "translate" if translate else "transcribe",
                "language": "english" if translate else (language or ""),
                "duration": len(samples) / rate,
                "text": result.text,
                "segments": [{"id": i, "seek": 0, "start": s.start, "end": s.end, "text": s.text}
                             for i, s in enumerate(result.segments)]})
        else:
            self._send(200, _subtitles(result.segments, vtt=fmt == "vtt").encode("utf-8"),
                       "text/vtt; charset=utf-8" if fmt == "vtt" else "application/x-subrip; charset=utf-8")
        return False

    @staticmethod
    def _translate(model, samples, language, timestamps):
        """Into English, by whichever means the file has: Whisper's `translate` task, or a target
        language (Canary). The engine refuses the one a file lacks before it decodes anything."""
        from . import LoomError
        try:
            return model.transcribe(samples, language=language, task="translate", timestamps=timestamps)
        except (LoomError, ValueError) as task_error:
            try:
                return model.transcribe(samples, language=language, target_language="en", timestamps=timestamps)
            except (LoomError, ValueError) as target_error:
                raise ApiError(400, f"this model does not translate: {task_error}; {target_error}",
                               param="model") from None

    def _translations(self):
        return self._transcriptions(translate=True)


def _driver_reads(model: Any, name: str) -> bool:
    """Whether the model's driver reads `inputs.<name>`: an input it does not read is ignored, and a
    request that names one would get the default it did not ask for."""
    source = getattr(model, "driver_source", "") or ""
    return re.search(rf"inputs\s*(?:\.\s*{name}\b|\[\s*['\"]{name}['\"]\s*\])", source) is not None


# -- the command --------------------------------------------------------------------------------------

def _load(spec: str, device: str) -> tuple[str, Any]:
    """`[NAME=]PATH` or `[NAME=]REPO_ID[:FILENAME]` (a Hugging Face repo, when no such file exists)."""
    from . import Model
    name, _, target = spec.rpartition("=") if "=" in spec else ("", "", spec)
    path = Path(target).expanduser()
    if path.is_file():
        return name or path.stem, Model.from_file(path, device=device)
    repo, _, filename = target.partition(":")
    if repo.count("/") != 1:
        raise SystemExit(f"loom-serve: {target!r} is neither a file nor a Hugging Face repo id (org/name)")
    return name or repo.split("/")[1], Model.from_pretrained(repo, filename=filename or None, device=device)


def make_server(models: Mapping[str, Any], host: str = "127.0.0.1", port: int = 8000, *,
                api_key: str | None = None, max_tokens: int = 512, quiet: bool = False) -> Server:
    """A server over `models` ({name: loom.Model}), bound and not yet serving: call `serve_forever()`.
    `port=0` picks a free port (`server.server_address`)."""
    return Server((host, port), Registry(models), api_key=api_key, max_tokens=max_tokens, quiet=quiet)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="loom-serve", description="Serve loom models behind OpenAI's API (chat, completions, "
        "speech, transcriptions).")
    parser.add_argument("models", nargs="+", metavar="MODEL",
                        help="[NAME=]PATH.gguf, or [NAME=]ORG/REPO[:FILE.gguf] from the Hugging Face Hub. "
                        "NAME is what requests put in `model`; the file's stem (the repo's name) by default.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="", help="as Model.from_file: '', cpu, gpu, npu or a device name")
    parser.add_argument("--api-key", default=os.environ.get("LOOM_API_KEY"),
                        help="require `Authorization: Bearer KEY` (default $LOOM_API_KEY; none: open)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="a chat completion's token cap when the request names none (default 512)")
    parser.add_argument("--quiet", action="store_true", help="no request log")
    args = parser.parse_args(argv)
    models: dict[str, Any] = {}
    for spec in args.models:
        name, model = _load(spec, args.device)
        if name in models:
            raise SystemExit(f"loom-serve: two models are named {name!r}; name one NAME=...")
        models[name] = model
    server = make_server(models, args.host, args.port, api_key=args.api_key, max_tokens=args.max_tokens,
                         quiet=args.quiet)
    host, port = server.server_address[:2]
    for s in server.registry.served.values():
        print(f"loom-serve: {s.name}: {s.interface or 'undeclared'} -> {DOORS.get(s.interface, 'no endpoint')}",
              file=sys.stderr)
    print(f"loom-serve: listening on http://{host}:{port}/v1", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
