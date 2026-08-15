"""The Python layer, without a model.

Everything here runs in seconds against the built extension and no GGUF, which is what makes it CI:
the questions are about argument coercion and error messages, and those are exactly the things that
are wrong in a way nobody notices until a wrong number reaches a model.

The half that needs a real model is `tests/gate/`.
"""
import builtins
import os
import sys
import tempfile
import types

import pytest

import loom


class TestPackage:
    def test_it_imports_and_exports_what_it_says(self):
        assert loom.__all__ == [
            "Model",
            "Tokenizer",
            "Transcription",
            "Segment",
            # The end-to-end layer. `Audio` and the three implemented interfaces are exported because
            # a caller annotates against them; the thirteen not-yet-implemented ones are reached as
            # `model.<name>` and are deliberately not names to import.
            "Audio",
            "Interface",
            "UnsupportedTask",
            "Text2Text",
            "Speech2Text",
            "Text2Speech",
            "LoomError",
            "devices",
            "download",
            "__version__",
        ]
        assert isinstance(loom.__version__, str)

    def test_loom_error_is_a_runtime_error(self):
        """The engine's own exception type reaches Python as itself, not flattened to RuntimeError --
        a caller distinguishing "this GGUF is wrong" from "this binding is wrong" needs the type."""
        assert issubclass(loom.LoomError, RuntimeError)

    def test_a_missing_file_says_so_before_the_engine_sees_it(self):
        with pytest.raises(FileNotFoundError, match="no such model file"):
            loom.Model.from_file("/definitely/not/here.gguf")


class TestInputCoercion:
    """`_as_value` decides what a driver receives. Every rejection here is a value that WOULD have
    converted silently to a number and produced a plausible, wrong answer."""

    def test_numbers_pass_through(self):
        assert loom._as_value("n_steps", 4) == 4.0
        assert loom._as_value("scale", 0.5) == 0.5

    def test_sequences_become_lists_of_floats(self):
        assert loom._as_value("tokens", [1, 2, 3]) == [1.0, 2.0, 3.0]
        assert loom._as_value("tokens", (1, 2)) == [1.0, 2.0]
        assert loom._as_value("waveform", range(3)) == [0.0, 1.0, 2.0]

    def test_anything_with_a_float_works_without_numpy_being_required(self):
        """The point of coercing elementwise: array.array, numpy arrays and torch tensors all satisfy
        this, and none of them is a dependency of a package whose job is handing arrays to C++."""
        import array
        assert loom._as_value("waveform", array.array("f", [0.5, -0.5])) == [0.5, -0.5]

    def test_a_bool_is_rejected_rather_than_becoming_one_point_zero(self):
        with pytest.raises(TypeError, match="bool"):
            loom._as_value("enable", True)

    def test_a_string_is_rejected_and_says_where_tokenisation_happens(self):
        with pytest.raises(TypeError, match="Tokenisation happens inside the model"):
            loom._as_value("text", "hello")

    def test_a_sequence_of_non_numbers_names_the_input(self):
        with pytest.raises(TypeError, match="tokens"):
            loom._as_value("tokens", [1, "two"])

    def test_a_mapping_or_set_is_rejected_for_having_no_order(self):
        """Both are iterable, so without an explicit check a dict marshals as its KEYS and a set as
        its elements in whatever order it held them -- a bug that shows up as bad output."""
        with pytest.raises(TypeError, match="no order to marshal"):
            loom._as_value("thing", {"a": 1})
        with pytest.raises(TypeError, match="no order to marshal"):
            loom._as_value("thing", {1, 2, 3})

    def test_an_unconvertible_object_names_its_type(self):
        with pytest.raises(TypeError, match="object"):
            loom._as_value("thing", object())


class _FakeHub(types.ModuleType):
    """Enough `huggingface_hub` to exercise the file-picking rule without a network."""

    def __init__(self, files, name="huggingface_hub"):
        super().__init__(name)
        self._files = files
        self.downloaded = None

    def list_repo_files(self, repo_id, revision=None, token=None):
        return self._files

    def hf_hub_download(self, repo_id, filename, revision=None, cache_dir=None, token=None):
        self.downloaded = filename
        return f"/cache/{repo_id}/{filename}"


class TestHubFileSelection:
    def _with_hub(self, monkeypatch, files):
        fake = _FakeHub(files)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
        return fake

    def test_a_single_gguf_needs_no_filename(self, monkeypatch):
        fake = self._with_hub(monkeypatch, ["README.md", "model.gguf"])
        assert str(loom.download("some/repo")).endswith("model.gguf")
        assert fake.downloaded == "model.gguf"

    def test_several_ggufs_raise_and_list_them_rather_than_guessing(self, monkeypatch):
        """Picking one is how somebody ends up reporting numbers for a quantisation they did not
        choose and cannot tell they got."""
        self._with_hub(monkeypatch, ["q8_0.gguf", "f32.gguf"])
        with pytest.raises(ValueError) as raised:
            loom.download("some/repo")
        assert "f32.gguf" in str(raised.value) and "q8_0.gguf" in str(raised.value)

    def test_no_gguf_at_all_says_so(self, monkeypatch):
        self._with_hub(monkeypatch, ["README.md"])
        with pytest.raises(FileNotFoundError, match="contains no .gguf"):
            loom.download("some/repo")

    def test_an_explicit_filename_skips_the_listing(self, monkeypatch):
        fake = self._with_hub(monkeypatch, ["q8_0.gguf", "f32.gguf"])
        loom.download("some/repo", filename="q8_0.gguf")
        assert fake.downloaded == "q8_0.gguf"

    def test_a_missing_hub_says_how_to_get_it_and_that_local_files_do_not_need_it(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("no module named huggingface_hub")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        with pytest.raises(ImportError) as raised:
            loom.download("some/repo")
        assert "pip install" in str(raised.value)
        assert "from_file" in str(raised.value)


class _FakeHandle:
    """A stand-in for the pybind11 `Model`, so the Python layer's decisions can be tested without a
    GGUF. Only the methods `loom.Model` actually calls are here."""

    def __init__(self, returns, vocab="gpt2", eos=-1, contract=None):
        self._returns = list(returns)      # what successive infer() calls hand back
        self._vocab = vocab
        self._eos = eos
        self.calls = []                    # every inputs dict infer() was given
        self.langs = []                    # every `lang` encode() was given
        self.generate_calls = []           # every argument set generate() was given
        self._contract = dict(contract or {})

    def contract(self):
        """What the file declares. Empty-but-present by default, which is a pre-contract GGUF -- the
        shape most models on disk still have, and the one the interface layer has to handle."""
        return dict(self._contract)

    def generate(self, tokens, max_new_tokens, eos_token, extra_inputs):
        """The engine's LM loop, which this double CANNOT stand in for -- and does not try to. It
        records what it was handed and replays a canned answer, so the tests below are about the
        marshalling either side of the call. The loop itself is pinned in loom.cpp
        (tests/ci/test_text_generate.cpp), where it now lives; a Python double asserting on loop
        behaviour would be asserting about a reimplementation of the thing under test."""
        self.generate_calls.append(
            {"tokens": list(tokens), "max_new_tokens": max_new_tokens, "eos_token": eos_token,
             "extra_inputs": dict(extra_inputs)})
        return self._returns.pop(0) if self._returns else []

    def has_tokenizer(self): return self._vocab is not None
    def tokenizer_kind(self): return self._vocab or ""
    def tokenizer_size(self): return 100
    def tokenizer_default_lang(self): return "en" if self._vocab == "supertonic" else ""

    def encode(self, text, lang=""):
        self.langs.append(lang)
        return [10, 11, 12]
    def decode(self, ids): return "|".join(str(i) for i in ids)
    def kv_i32(self, key, fallback): return self._eos
    def device_name(self): return "CPU"
    def device_description(self): return "a fake device"
    def call(self, fn_name, inputs):
        self.calls.append(inputs)
        return self._returns.pop(0) if self._returns else 0.0
    # `Model.transcribe` runs the engine's whole long-form loop, which a fake cannot stand in for --
    # it needs a real vocabulary and a real driver. What a double CAN pin is the marshalling either
    # side of it, so this returns the shape the binding returns and nothing more.
    def transcribe(self, waveform, options):
        self.calls.append({"waveform": waveform, **options})
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello", "closed": True}],
                "text": "hello", "windows": 1, "timestamped": True}


def _model(handle):
    from pathlib import Path
    return loom.Model(handle, Path("fake.gguf"))


class TestTokenizer:
    def test_a_model_without_a_vocabulary_reports_none_rather_than_a_broken_object(self):
        """The phoneme-input TTS families embed no vocabulary -- they take ids from outside the
        engine -- so None is the honest answer, not an object whose every method raises."""
        assert _model(_FakeHandle([], vocab=None)).tokenizer is None

    def test_a_model_with_one_exposes_its_kind_and_size(self):
        tok = _model(_FakeHandle([])).tokenizer
        assert tok.kind == "gpt2" and tok.size == 100
        assert "gpt2" in repr(tok)

    def test_tokenize_and_detokenize_round_trip_through_the_model(self):
        model = _model(_FakeHandle([]))
        assert model.tokenize("hello") == [10, 11, 12]
        # floats, because floats are what infer() returns
        assert model.detokenize([1.0, 2.0]) == "1|2"

    def test_an_omitted_lang_reaches_the_engine_as_empty_not_as_a_guess(self):
        """The engine substitutes the file's own declared default for an empty lang. This layer must
        not invent one on its way there -- a hardcoded "en" here would silently override an export
        that declared something else."""
        handle = _FakeHandle([])
        _model(handle).tokenize("hello")
        assert handle.langs == [""]

    def test_a_named_lang_is_passed_through(self):
        handle = _FakeHandle([], vocab="supertonic")
        _model(handle).tokenize("hello", lang="ko")
        assert handle.langs == ["ko"]

    def test_a_language_tagged_vocabulary_says_which_language_it_defaults_to(self):
        tok = _model(_FakeHandle([], vocab="supertonic")).tokenizer
        assert tok.default_lang == "en"
        assert "lang='en'" in repr(tok)

    def test_a_vocabulary_with_no_language_concept_says_so_rather_than_inventing_one(self):
        tok = _model(_FakeHandle([])).tokenizer
        assert tok.default_lang == ""
        assert "lang=" not in repr(tok), "an empty default must not print as lang=''"


class TestGenerateMarshalsAroundTheEngineLoop:
    """The loop that tells a sequence-returning driver from a single-token one is the ENGINE's now
    (`loom::text::generate`), and its behaviour is pinned in loom.cpp rather than here. It was a Python
    loop, correct, and still a second copy: loom_cli's differed in three ways at once -- no eos stop,
    the first element of a list return rather than the last, and a silent id clamp -- with nothing but
    coincidence keeping them in step.

    What stays this layer's job, and is what these check: encode/decode either side, and handing the
    engine a request that says what the caller meant."""

    def test_generate_encodes_and_decodes_around_the_call(self):
        handle = _FakeHandle([[7, 8]])
        assert _model(handle).generate("anything") == "7|8"
        assert handle.generate_calls[0]["tokens"] == [10, 11, 12], "the encoded prompt, not the string"

    def test_no_eos_named_asks_the_file_rather_than_disabling_the_stop(self):
        """-2 is 'ask the file', -1 is 'do not stop early'. One sentinel could not carry both, and
        collapsing them would turn 'I did not specify' into 'run to the ceiling' -- which is how the
        CLI's copy of this loop behaved, for exactly that reason."""
        handle = _FakeHandle([[1]])
        _model(handle).generate_ids([1], max_new_tokens=5)
        assert handle.generate_calls[0]["eos_token"] == -2

    def test_an_explicit_eos_is_passed_as_given_including_the_disabling_one(self):
        handle = _FakeHandle([[1], [1]])
        model = _model(handle)
        model.generate_ids([1], eos_token=99)
        model.generate_ids([1], eos_token=-1)
        assert [c["eos_token"] for c in handle.generate_calls] == [99, -1]

    def test_extra_driver_inputs_are_forwarded(self):
        """A model whose `infer` takes more than tokens -- a style vector, a speaker id -- is driven
        through the same loop, so the extras have to survive the trip."""
        handle = _FakeHandle([[1]])
        _model(handle).generate_ids([1], style=[0.5, 0.25], speaker=3)
        extras = handle.generate_calls[0]["extra_inputs"]
        assert extras == {"style": [0.5, 0.25], "speaker": 3.0}


class TestDeviceIsPassedThroughAndReadBack:
    """Where a model runs, at the Python layer. What device spec resolves to WHICH device is the
    engine's decision and is pinned there (loom.cpp tests/ci/test_device_selection.cpp); what is
    decided here is only that a caller's choice reaches it and that the answer comes back -- which is
    the half that would break silently, since a dropped `device=` argument still loads the model.
    """

    def test_the_spec_reaches_the_extension_and_defaults_to_deciding_for_itself(self, monkeypatch, tmp_path):
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"")
        seen = []
        monkeypatch.setattr(loom._loom, "Model",
                            lambda path, device: seen.append((path, device)) or _FakeHandle([]))
        loom.Model.from_file(gguf)
        loom.Model.from_file(gguf, device="cpu")
        # An empty default rather than "cpu": a wheel built with a GPU backend should use it without
        # every caller having to ask, and one built without has only a CPU for "" to resolve to.
        assert [d for _, d in seen] == ["", "cpu"]

    def test_from_pretrained_forwards_it_too(self, monkeypatch, tmp_path):
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"")
        monkeypatch.setattr(loom, "download", lambda *a, **k: gguf)
        seen = []
        monkeypatch.setattr(loom._loom, "Model",
                            lambda path, device: seen.append(device) or _FakeHandle([]))
        loom.Model.from_pretrained("org/repo", device="Vulkan0")
        assert seen == ["Vulkan0"]

    def test_the_resolved_device_is_readable(self):
        handle = _FakeHandle([])
        model = _model(handle)
        assert model.device == "CPU"
        assert model.device_description == "a fake device"


# The declared contract a Whisper-like export writes, as the interface layer sees it.
_ASR_CONTRACT = {
    "declared": True, "task": "automatic-speech-recognition", "input_kind": "audio",
    "output_kind": "token_ids", "interface": "speech2text", "sample_rate": 16000,
    "clip_samples": 480000, "max_input_tokens": 0, "text_frontend": "vocab",
    "phoneme_alphabet": "", "phonemizer_ruleset": "", "languages": ["en"], "entry_points": ["infer"],
    "default_steps": 0, "voices": [],
}
_TTS_PHONEME_CONTRACT = dict(_ASR_CONTRACT, task="text-to-speech", input_kind="phoneme_ids",
                             output_kind="audio", interface="text2speech", text_frontend="",
                             phoneme_alphabet="ipa", sample_rate=22050, clip_samples=0,
                             default_steps=10)
_TTS_TEXT_CONTRACT = dict(_TTS_PHONEME_CONTRACT, input_kind="text", text_frontend="vocab")


class TestInterfacesAreTheModalityPair:
    """Which door a model answers to is read off its declared contract, never off its architecture.

    That is the whole reason this layer can exist inside a package whose standing rule is that it holds
    no per-architecture code: `Text2Speech` is not a category invented here, it is the I/O contract, and
    the file states it."""

    def test_the_declared_pair_selects_the_interface(self):
        model = _model(_FakeHandle([], contract=_ASR_CONTRACT))
        assert model.task == "automatic-speech-recognition"
        assert model.capabilities == ("speech2text",)
        assert model.speech2text.supported
        assert not model.text2speech.supported

    def test_every_interface_is_present_even_when_it_raises(self):
        """A missing method answers 'no such thing'; a present one that raises answers 'not this
        model, and here is what it is'. The second is what a caller probing capabilities needs."""
        model = _model(_FakeHandle([], contract=_ASR_CONTRACT))
        assert hasattr(model, "text2image") and hasattr(model, "image2segmentationmask")
        with pytest.raises(loom.UnsupportedTask) as excinfo:
            model.text2speech.infer("hello")
        assert "speech2text" in str(excinfo.value), "the error must name what the model actually is"

    def test_an_undeclared_file_offers_no_door_and_says_why(self):
        """Every GGUF exported before the contract existed lands here. Guessing from its architecture
        is exactly the per-architecture code this layer refuses to contain, so it offers nothing --
        and `infer` is untouched, which is what such a file does support."""
        model = _model(_FakeHandle([], contract={}))
        assert model.task == "" and model.capabilities == ()
        with pytest.raises(loom.UnsupportedTask) as excinfo:
            model.text2text.infer("hello")
        assert "declares no task" in str(excinfo.value)
        assert "model.infer" in str(excinfo.value), "it must point at the door that does work"

    def test_text2text_is_the_same_call_as_generate(self):
        handle = _FakeHandle([[7, 8], [7, 8]],
                             contract=dict(_ASR_CONTRACT, task="text-generation", input_kind="text",
                                           output_kind="text", interface="text2text"))
        model = _model(handle)
        assert model.text2text.infer("anything") == model.generate("anything") == "7|8"


class TestText2Speech:
    def test_a_waveform_comes_back_with_its_rate_attached(self):
        """A bare list whose rate the caller has to remember is the same defect `Transcription` avoids
        one modality over: 24 kHz played at 22.05 kHz is not an error, it is a slightly deep voice."""
        handle = _FakeHandle([[0.0, 0.5, -0.5]], contract=_TTS_PHONEME_CONTRACT)
        audio = _model(handle).text2speech.infer(phonemes=[1, 2, 3])
        assert isinstance(audio, loom.Audio)
        assert audio.sample_rate == 22050 and len(audio) == 3
        assert audio.duration == pytest.approx(3 / 22050)

    def test_the_declared_step_default_is_applied_and_the_caller_still_wins(self):
        """A sampler step count is a property of the export. A host inventing one is how two front ends
        produce different audio from the same file."""
        handle = _FakeHandle([[0.0], [0.0]], contract=_TTS_PHONEME_CONTRACT)
        model = _model(handle)
        model.text2speech.infer(phonemes=[1])
        model.text2speech.infer(phonemes=[1], steps=4)
        assert [c["n_steps"] for c in handle.calls] == [10.0, 4.0]

    def test_a_phoneme_model_refuses_text_and_says_it_is_the_model_not_the_package(self):
        handle = _FakeHandle([], contract=_TTS_PHONEME_CONTRACT)
        with pytest.raises(loom.UnsupportedTask) as excinfo:
            _model(handle).text2speech.infer("hello")
        assert "ipa" in str(excinfo.value)
        assert "phonemes=" in str(excinfo.value), "it must name the door that does work"

    def test_a_grapheme_model_takes_text_directly(self):
        """Supertonic is not in the phoneme group -- it encodes graphemes itself. The distinction is
        per-model and declared, which is why no code here names either model."""
        handle = _FakeHandle([[0.0]], contract=_TTS_TEXT_CONTRACT)
        _model(handle).text2speech.infer("hello")
        assert handle.calls[0]["tokens"] == [10.0, 11.0, 12.0]

    def test_the_three_inputs_are_depths_not_alternatives(self):
        handle = _FakeHandle([], contract=_TTS_TEXT_CONTRACT)
        with pytest.raises(TypeError):
            _model(handle).text2speech.infer("hello", phonemes=[1])
        with pytest.raises(TypeError):
            _model(handle).text2speech.infer()


class TestAudio:
    def test_it_writes_a_wav_a_reader_can_open(self):
        import wave

        audio = loom.Audio(samples=[0.0, 1.0, -1.0, 0.5], sample_rate=16000)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.wav")
            audio.save(path)
            with wave.open(path, "rb") as f:
                assert f.getframerate() == 16000 and f.getnchannels() == 1
                assert f.getnframes() == 4

    def test_a_model_that_declared_no_rate_refuses_rather_than_guessing(self):
        """Writing 22.05 kHz audio at an assumed 16 kHz produces a file that plays, which is worse
        than one that does not."""
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError):
                loom.Audio(samples=[0.0], sample_rate=0).save(os.path.join(d, "x.wav"))
