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
import warnings
from unittest import mock

import pytest

import loom


class TestPackage:
    def test_it_imports_and_exports_what_it_says(self):
        assert loom.__all__ == [
            "Model",
            "Tokenizer",
            "Transcription",
            "Segment",
            # The end-to-end layer. The result types and the six implemented interfaces are exported
            # because a caller annotates against them; the eleven not-yet-implemented ones are reached
            # as `model.<name>` and are deliberately not names to import.
            "Audio",
            "Classification",
            "TokenClass",
            "Interface",
            "UnsupportedTask",
            "Text2Text",
            "Speech2Text",
            "Text2Speech",
            "Text2Class",
            "Text2Codes",
            "Codes2Speech",
            # The G2P frontend: a module rather than a class, because a caller registers into it.
            "phonemizers",
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

    def __init__(self, returns, vocab="gpt2", eos=-1, contract=None, transcribe_warnings=(),
                 chat_roles=(), hparams=None):
        self._returns = list(returns)      # what successive infer() calls hand back
        # What the ENGINE reports as ignored -- an argument this file has nothing to select with. The
        # engine returns these instead of printing them (a library has no logger); the Python layer is
        # what turns them into warnings, and that hand-off is what the test below pins.
        self._transcribe_warnings = tuple(transcribe_warnings)
        self._vocab = vocab
        # Empty by default: most GGUFs on disk carry no chat template, and the layer has to handle that
        # rather than assume every text model can hold a conversation.
        self._chat_roles = tuple(chat_roles)
        self.encode_calls = []
        self._eos = eos
        self.calls = []                    # every inputs dict infer() was given
        self.langs = []                    # every `lang` encode() was given
        self.generate_calls = []           # every argument set generate() was given
        self.classify_calls = []           # every argument set classify() was given
        self._contract = dict(contract or {})
        self._hparams = dict(hparams or {})

    # The binding exposes one reader PER GGUF TYPE (`hparam_u32`/`hparam_f32`/`hparam_str`), because
    # GGUF's own types are explicit and a key written as u32 and read as f32 is an error the engine
    # raises. `Model.hparam` picks the reader from its `kind` argument, so a double defining a single
    # `hparam` is never called at all -- which is how this fake first "passed" by raising the absent-key
    # error for a key it had been given.
    def _hparam(self, key):
        if key not in self._hparams:
            raise RuntimeError(f"no such hparam: {key}")
        return self._hparams[key]

    def hparam_u32(self, key): return self._hparam(key)
    def hparam_f32(self, key): return self._hparam(key)
    def hparam_str(self, key): return self._hparam(key)

    def contract(self):
        """What the file declares. Empty-but-present by default, which is a pre-contract GGUF -- the
        shape most models on disk still have, and the one the interface layer has to handle."""
        return dict(self._contract)

    def generate(self, tokens, max_new_tokens, eos_token, extra_inputs,
                 temperature, top_k, top_p, seed):
        """The engine's LM loop, which this double CANNOT stand in for -- and does not try to. It
        records what it was handed and replays a canned answer, so the tests below are about the
        marshalling either side of the call. The loop itself is pinned in loom.cpp
        (tests/ci/test_text_generate.cpp), where it now lives; a Python double asserting on loop
        behaviour would be asserting about a reimplementation of the thing under test.

        The four sampling arguments are positional and unconditional, matching the binding: `None` is
        the value that means "use what the file declared" (P4.24), so there is nothing to omit."""
        self.generate_calls.append(
            {"tokens": list(tokens), "max_new_tokens": max_new_tokens, "eos_token": eos_token,
             "extra_inputs": dict(extra_inputs), "temperature": temperature, "top_k": top_k,
             "top_p": top_p, "seed": seed})
        return self._returns.pop(0) if self._returns else []

    # -- the chat door (P4.23) --------------------------------------------------------------------
    def has_chat_template(self): return bool(self._chat_roles)
    def chat_roles(self): return list(self._chat_roles)

    def apply_chat_template(self, messages, add_generation_prompt):
        """A ChatML-shaped stand-in. Same caveat as `generate`: the real assembly is
        `loom::ChatTemplate` and is pinned in loom.cpp (tests/ci/test_chat_template.cpp); what these
        tests are about is that this package hands it the right conversation and does not render one
        itself."""
        if not self._chat_roles:
            raise RuntimeError("this GGUF carries no chat template")
        for role, _ in messages:
            if role not in self._chat_roles:
                raise RuntimeError(f"no '{role}' role")
        body = "".join(f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in messages)
        return body + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def eos_token_ids(self): return [7]

    def has_tokenizer(self): return self._vocab is not None
    def tokenizer_kind(self): return self._vocab or ""
    def tokenizer_size(self): return 100
    def tokenizer_default_lang(self): return "en" if self._vocab == "supertonic" else ""

    def encode(self, text, lang=""):
        self.langs.append(lang)
        self.encode_calls.append(text)
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
                "text": "hello", "windows": 1, "timestamped": True,
                "warnings": list(self._transcribe_warnings)}

    # Same reasoning as `transcribe`: the framing-token strip and the label lookup are the ENGINE's
    # (loom/core/text_classify.h), so what a double can pin is the marshalling either side. It records
    # what it was handed and returns one entry per token it was given.
    def classify(self, tokens, strip_special, extra_inputs):
        self.classify_calls.append({"tokens": list(tokens), "strip_special": strip_special,
                                    "extra_inputs": dict(extra_inputs)})
        return [{"token": int(t), "label_id": i % 2, "label": ["O", "B-PER"][i % 2]}
                for i, t in enumerate(tokens)]


class TestTranscribeWarnings:
    """An argument the engine ignored becomes a Python warning, not an exception and not silence.

    `language="en"` on a monolingual checkpoint used to RAISE, which sent callers looking for a defect
    in a pipeline that had none -- the argument named exactly what the model was always going to do.
    It is ignored now, and the engine says so. Refusing was wrong; staying quiet would be worse, since
    the caller clearly believed the argument did something.

    A request the model cannot SERVE -- a language a multilingual file lacks, `translate` on a file
    with no task tokens -- still raises, from the engine, and there is nothing for this layer to do.
    """

    def test_an_ignored_argument_is_raised_as_a_runtime_warning(self):
        handle = _FakeHandle([], transcribe_warnings=["language=\"en\" selects nothing here"])
        with pytest.warns(RuntimeWarning, match="selects nothing"):
            result = _model(handle).transcribe([0.0, 1.0], language="en")
        assert result.text == "hello", "the call still returns its transcript"

    def test_no_warning_when_the_engine_reports_none(self):
        handle = _FakeHandle([])
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # any warning at all fails here
            _model(handle).transcribe([0.0, 1.0], language="en")


def tmp_lexicon():
    """A two-entry `word<TAB>ipa` TSV on disk. Never read here -- orthography2ipa resolves a lexicon
    lazily, on the first transcription for that language -- but a real path is what a caller passes."""
    fd, path = tempfile.mkstemp(suffix=".tsv")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("time\ttˈaɪm\nfriend\tfɹˈɛnd\n")
    return path


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




class TestSamplingKnobsAreUnsetUntilNamed:
    """P4.24. The knobs reach the DRIVER, and the value that means "use what the checkpoint declared"
    is `None` -- not a number this layer picked.

    That distinction is the whole design: a Python-side default of, say, `temperature=1.0` would
    silently overrule every file for every caller who named nothing, and every byte-identity baseline
    with it. What the engine does with the knobs is pinned in loom.cpp (tests/ci/test_sample_row.cpp)."""

    def test_naming_nothing_passes_nothing(self):
        handle = _FakeHandle([[1]])
        _model(handle).generate_ids([1])
        call = handle.generate_calls[0]
        assert (call["temperature"], call["top_k"], call["top_p"], call["seed"]) == (None,) * 4

    def test_named_knobs_arrive_typed(self):
        handle = _FakeHandle([[1]])
        _model(handle).generate_ids([1], temperature=0.7, top_k=40, top_p=0.9, seed=11)
        call = handle.generate_calls[0]
        assert call["temperature"] == 0.7 and isinstance(call["temperature"], float)
        assert call["top_k"] == 40 and isinstance(call["top_k"], int)
        assert call["top_p"] == 0.9
        assert call["seed"] == 11

    def test_a_knob_is_not_a_driver_input(self):
        """`temperature` is a named argument, so it must not also land in the driver's extras -- the
        driver reads it from its own `inputs.temperature`, which the engine fills in."""
        handle = _FakeHandle([[1]])
        _model(handle).generate_ids([1], temperature=0.7, style=[0.5])
        assert handle.generate_calls[0]["extra_inputs"] == {"style": [0.5]}


class TestChatIsGenerateWithTheTemplateApplied:
    """P4.23. This layer's job is to hand the ENGINE a conversation and the engine's own assembly back
    to `generate`; the template itself is data in the GGUF and the assembly is `loom::ChatTemplate`
    (pinned in loom.cpp, tests/ci/test_chat_template.cpp). Nothing here renders Jinja or carries a
    per-model string, which is the point of the split."""

    def test_a_bare_string_is_the_user_turn(self):
        handle = _FakeHandle([[7, 8]], chat_roles=("user", "assistant"))
        assert _model(handle).chat("hi") == "7|8"
        # The prompt reached `generate` templated, not raw: the fake encodes to a fixed id list, so
        # what is asserted is that the model asked the handle to template it at all.
        assert handle.encode_calls[-1].startswith("<|im_start|>user\n")
        assert handle.encode_calls[-1].endswith("<|im_start|>assistant\n")

    def test_a_conversation_may_be_pairs_or_dicts(self):
        """Both, because a caller moving code across from `transformers` should not have to rewrite
        the data."""
        pairs = _FakeHandle([[1]], chat_roles=("user", "assistant"))
        dicts = _FakeHandle([[1]], chat_roles=("user", "assistant"))
        _model(pairs).chat([("user", "a"), ("assistant", "b"), ("user", "c")])
        _model(dicts).chat([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                            {"role": "user", "content": "c"}])
        assert pairs.encode_calls[-1] == dicts.encode_calls[-1]

    def test_a_model_with_no_template_says_so(self):
        handle = _FakeHandle([[1]])
        assert _model(handle).chat_roles == []
        with pytest.raises(RuntimeError, match="no chat template"):
            _model(handle).chat("hi")

    def test_a_role_the_checkpoint_does_not_declare_is_an_error(self):
        """Gemma 3 is the live case: its template folds a system message into the first user turn
        rather than emitting a block for it, so it declares two roles and a system message must be
        refused rather than dropped."""
        handle = _FakeHandle([[1]], chat_roles=("user", "assistant"))
        assert _model(handle).chat_roles == ["user", "assistant"]
        with pytest.raises(RuntimeError, match="'system'"):
            _model(handle).chat([("system", "be terse"), ("user", "hi")])

    def test_text2text_chat_is_the_same_call(self):
        via_model = _FakeHandle([[7, 8]], chat_roles=("user", "assistant"),
                                contract=dict(task="text-generation", input_kind="text",
                                              output_kind="text", interface="text2text"))
        via_interface = _FakeHandle([[7, 8]], chat_roles=("user", "assistant"),
                                    contract=dict(task="text-generation", input_kind="text",
                                                  output_kind="text", interface="text2text"))
        assert _model(via_model).chat("hi") == _model(via_interface).text2text.chat("hi") == "7|8"

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
    "default_steps": 0, "voices": [], "labels": [],
}
_TTS_PHONEME_CONTRACT = dict(_ASR_CONTRACT, task="text-to-speech", input_kind="phoneme_ids",
                             output_kind="audio", interface="text2speech", text_frontend="",
                             phoneme_alphabet="ipa", sample_rate=22050, clip_samples=0,
                             default_steps=10)
_TTS_TEXT_CONTRACT = dict(_TTS_PHONEME_CONTRACT, input_kind="text", text_frontend="vocab")
_TOKEN_CLASS_CONTRACT = dict(_ASR_CONTRACT, task="token-classification", input_kind="text",
                             output_kind="class", interface="text2class", sample_rate=0,
                             clip_samples=0, labels=["O", "B-PER"])
_CODEC_CONTRACT = dict(_ASR_CONTRACT, task="audio-codec", input_kind="audio_codes",
                       output_kind="audio", interface="codes2speech", sample_rate=44100,
                       clip_samples=0, text_frontend="")
_CODES_LM_CONTRACT = dict(_ASR_CONTRACT, task="text-to-codes", input_kind="text",
                          output_kind="audio_codes", interface="text2codes", sample_rate=0,
                          clip_samples=0, text_frontend="vocab")


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

    def test_a_declared_classifier_answers_text2class(self):
        """The first non-audio pair any family declares, and the reason `Text2Class` stopped being a
        planned interface: nothing here learned an architecture, the file simply said `class`."""
        model = _model(_FakeHandle([], contract=_TOKEN_CLASS_CONTRACT))
        assert model.task == "token-classification"
        assert model.capabilities == ("text2class",)
        assert not model.text2text.supported

    def test_text2class_encodes_and_labels(self):
        handle = _FakeHandle([], contract=_TOKEN_CLASS_CONTRACT)
        result = _model(handle).text2class.infer("hello there")
        # Encoded through the model's own vocabulary -- the fake returns [10, 11, 12] for any text.
        assert handle.encode_calls == ["hello there"]
        assert handle.classify_calls[0]["tokens"] == [10, 11, 12]
        assert [t.label for t in result] == ["O", "B-PER", "O"]
        # The label SET travels with the labels, because a caller checking which classes existed
        # cannot recover it from the ones that happened to be chosen.
        assert result.labels == ["O", "B-PER"]
        # And the pieces, which are not recoverable afterwards: a WordPiece encode splits words, so
        # joining them back into words is a rule this layer has no basis to make for the caller.
        assert [t.piece for t in result] == ["10", "11", "12"]

    def test_ids_may_be_passed_instead_of_text(self):
        handle = _FakeHandle([], contract=_TOKEN_CLASS_CONTRACT)
        _model(handle).text2class.infer(tokens=[5, 6])
        assert handle.encode_calls == [], "ids must not be re-encoded"
        assert handle.classify_calls[0]["tokens"] == [5, 6]

    def test_text_and_tokens_are_two_depths_of_one_input_not_two_arguments(self):
        model = _model(_FakeHandle([], contract=_TOKEN_CLASS_CONTRACT))
        with pytest.raises(TypeError):
            model.text2class.infer("hello", tokens=[1])
        with pytest.raises(TypeError):
            model.text2class.infer()

    def test_strip_special_reaches_the_engine_as_the_engines_decision(self):
        """The knob is a switch on a decision the ENGINE makes, so what this layer owes is to pass it
        through -- and to default it the same way the engine's own default reads."""
        handle = _FakeHandle([], contract=_TOKEN_CLASS_CONTRACT)
        model = _model(handle)
        model.text2class.infer(tokens=[1])
        model.text2class.infer(tokens=[1], strip_special=False)
        assert [c["strip_special"] for c in handle.classify_calls] == [True, False]

    def test_a_classifier_with_no_vocabulary_says_so_rather_than_guessing(self):
        handle = _FakeHandle([], vocab=None, contract=_TOKEN_CLASS_CONTRACT)
        with pytest.raises(loom.UnsupportedTask, match="embeds no vocabulary"):
            _model(handle).text2class.infer("hello")

    def test_an_ar_codec_lm_answers_text2codes_and_returns_frames(self):
        """The first half of the family-10 pair. `audio_codes` as an OUTPUT kind is what makes this a
        door of its own rather than `text2speech` -- the model produces something a codec turns into
        audio, and ADR-022 keeps that a second file."""
        handle = _FakeHandle([[1, 2, 3, 4, 5, 6]], contract=_CODES_LM_CONTRACT,
                             hparams={"codec.n_codebooks": 3})
        model = _model(handle)
        assert model.capabilities == ("text2codes",)
        assert not model.text2speech.supported
        # Frame-major rows, which is the layout `Codes2Speech` takes and the one the width can be
        # checked against on the way in.
        assert model.text2codes.infer("hello") == [[1, 2, 3], [4, 5, 6]]

    def test_text2codes_output_feeds_codes2speech_unchanged(self):
        """The composition, spelled as a host writes it: two files, two calls, the array between them.

        This is the assertion ADR-022 costs -- with the codec in a second GGUF, nothing inside either
        one says the pair fits, so the layouts have to be pinned where they meet. The engine-side
        version of this against the real checkpoints is loom.cpp's
        `test_e2e_dia_dac_composition.cpp`."""
        lm = _FakeHandle([[1, 2, 3, 4, 5, 6]], contract=_CODES_LM_CONTRACT,
                         hparams={"codec.n_codebooks": 3})
        codec = _FakeHandle([[0.1, 0.2]], contract=_CODEC_CONTRACT,
                            hparams={"codec.n_codebooks": 3})
        codes = _model(lm).text2codes.infer("hello")
        audio = _model(codec).codes2speech.infer(codes)
        assert codec.calls[0]["codes"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert audio.sample_rate == 44100

    def test_max_new_tokens_reaches_the_driver_only_when_named(self):
        """It counts AUDIO FRAMES, and the default belongs to the file. A host that always passed one
        would override a ceiling the export derived from the model's own position budget."""
        named = _FakeHandle([[1, 2, 3]], contract=_CODES_LM_CONTRACT,
                            hparams={"codec.n_codebooks": 3})
        unnamed = _FakeHandle([[1, 2, 3]], contract=_CODES_LM_CONTRACT,
                              hparams={"codec.n_codebooks": 3})
        _model(named).text2codes.infer("hello", max_new_tokens=16)
        _model(unnamed).text2codes.infer("hello")
        assert named.calls[0]["max_new_tokens"] == 16.0
        assert "max_new_tokens" not in unnamed.calls[0]

    def test_a_partial_final_frame_is_the_export_disagreeing_with_the_file(self):
        """Seven codes at three codebooks is not a caller error and must not be silently truncated --
        it is the driver and the declared width disagreeing, which downstream is audio of the wrong
        duration and nothing raised."""
        handle = _FakeHandle([[1, 2, 3, 4, 5, 6, 7]], contract=_CODES_LM_CONTRACT,
                             hparams={"codec.n_codebooks": 3})
        with pytest.raises(ValueError, match="whole number of frames"):
            _model(handle).text2codes.infer("hello")

    def test_a_codes_lm_that_declares_no_width_says_so(self):
        """Absent is an export too old to state it, which is a different thing from a model with no
        codebooks -- so it is caught rather than defaulted to a guess about the frame width."""
        handle = _FakeHandle([[1, 2, 3]], contract=_CODES_LM_CONTRACT)
        with pytest.raises(loom.UnsupportedTask, match="codec.n_codebooks"):
            _model(handle).text2codes.infer("hello")

    def test_text2codes_takes_text_or_ids_but_not_both(self):
        handle = _FakeHandle([[1, 2, 3]], contract=_CODES_LM_CONTRACT,
                             hparams={"codec.n_codebooks": 3})
        model = _model(handle)
        with pytest.raises(TypeError):
            model.text2codes.infer("hello", tokens=[1])
        with pytest.raises(TypeError):
            model.text2codes.infer()

    def test_a_declared_codec_answers_codes2speech(self):
        """`audio_codes` does NOT fold onto "text" -- ADR-020. A codec declaring `token_ids` would
        land on `text2speech` and be offered a text door it has no vocabulary for, which is the
        failure this interface exists to make impossible."""
        model = _model(_FakeHandle([[0.1, 0.2]], contract=_CODEC_CONTRACT,
                                    hparams={"codec.n_codebooks": 2}))
        assert model.task == "audio-codec"
        assert model.capabilities == ("codes2speech",)
        assert not model.text2speech.supported

    def test_codes_are_accepted_as_rows_or_flat(self):
        """Two spellings of one input, because both are what a caller actually holds: a driver that
        emitted them hands over a flat run, and a person writing them out writes rows."""
        rows = _FakeHandle([[0.1]], contract=_CODEC_CONTRACT, hparams={"codec.n_codebooks": 2})
        flat = _FakeHandle([[0.1]], contract=_CODEC_CONTRACT, hparams={"codec.n_codebooks": 2})
        _model(rows).codes2speech.infer([[1, 2], [3, 4]])
        _model(flat).codes2speech.infer([1, 2, 3, 4])
        assert rows.calls[0]["codes"] == flat.calls[0]["codes"] == [1.0, 2.0, 3.0, 4.0]

    def test_a_flat_run_that_is_not_whole_frames_is_refused(self):
        """Silently reinterpreting it as a different frame count produces audio of the wrong duration
        and no error anywhere, which is the one failure mode a codec caller cannot see."""
        handle = _FakeHandle([[0.1]], contract=_CODEC_CONTRACT, hparams={"codec.n_codebooks": 2})
        with pytest.raises(ValueError, match="whole number of frames"):
            _model(handle).codes2speech.infer([1, 2, 3])

    def test_a_row_of_the_wrong_width_names_the_rows(self):
        handle = _FakeHandle([[0.1]], contract=_CODEC_CONTRACT, hparams={"codec.n_codebooks": 2})
        with pytest.raises(ValueError, match="2 codebooks per frame"):
            _model(handle).codes2speech.infer([[1, 2], [3, 4, 5]])

    def test_the_declared_rate_travels_with_the_samples(self):
        handle = _FakeHandle([[0.1, 0.2]], contract=_CODEC_CONTRACT,
                             hparams={"codec.n_codebooks": 2})
        audio = _model(handle).codes2speech.infer([[1, 2]])
        assert audio.sample_rate == 44100, "the file's own rate, not the fallback"

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

    def test_a_model_with_no_text_front_end_refuses_text(self):
        """Declaring nothing is different from declaring `phonemes`: the first cannot encode text at
        all, and says so pointing at the doors that do work."""
        handle = _FakeHandle([], contract=dict(_TTS_PHONEME_CONTRACT, text_frontend=""))
        with pytest.raises(loom.UnsupportedTask) as excinfo:
            _model(handle).text2speech.infer("hello")
        assert "phonemes=" in str(excinfo.value) and "tokens=" in str(excinfo.value)

    def test_a_phoneme_model_phonemizes_then_encodes_with_its_own_table(self):
        """Two steps, and only the second is in the file. G2P is a property of the LANGUAGE so it lives
        outside every GGUF; the symbol table that turns its output into this checkpoint's ids is the
        checkpoint's own and now travels with it."""
        handle = _FakeHandle([[0.0]], contract=dict(_TTS_PHONEME_CONTRACT, text_frontend="phonemes"))
        seen = []
        loom.phonemizers.register("ipa", lambda text, language: seen.append((text, language)) or "hɛ")
        try:
            _model(handle).text2speech.infer("hello", language="en")
        finally:
            loom.phonemizers._PROVIDERS.pop("ipa", None)
        assert seen == [("hello", "en")], "the raw text reaches the phonemizer, not ids"
        assert handle.calls[0]["tokens"] == [10.0, 11.0, 12.0], "its output is encoded by the model"

    def test_a_lexicon_is_named_once_and_stored_under_the_resolved_language(self):
        """`set_lexicon` exists because orthography2ipa cannot reach English by rule -- "time" is `tɪm`
        without one -- and because no PARAMETER fixes that: `search="beam"` returns the greedy string
        unchanged at every width. It is stored rather than passed per call because registration in that
        library is process-global and lazily resolved.

        The resolution is the part worth pinning. A lexicon registered under an unresolved tag is
        SILENT when it is wrong -- nothing raises, the overlay simply never loads -- so `"en"` and the
        `"en-GB"` it resolves to must land in the same slot.
        """
        o2i = pytest.importorskip("orthography2ipa")
        tsv = tmp_lexicon()
        try:
            loom.phonemizers.set_lexicon(tsv, language="en")
            assert loom.phonemizers.lexicons() == {o2i.resolve("en"): str(tsv)}
            loom.phonemizers.set_lexicon(tsv, language="en-GB")
            assert len(loom.phonemizers.lexicons()) == 1, "one slot, not one per spelling"
            loom.phonemizers.set_lexicon(None)
            assert loom.phonemizers.lexicons() == {}, "None clears rather than registering 'None'"
        finally:
            loom.phonemizers._LEXICONS.clear()

    def test_a_lexicon_without_the_default_provider_installed_is_refused(self):
        """Accepting it would be worse than refusing: a lexicon set on a provider that does not exist
        is stored, never applied, and the caller hears unchanged audio with nothing to explain it."""
        with mock.patch.dict(sys.modules, {"orthography2ipa": None}):
            with pytest.raises(LookupError) as excinfo:
                loom.phonemizers.set_lexicon("/does/not/matter.tsv")
        assert "phonemes" in str(excinfo.value)

    def test_a_phoneme_string_is_encoded_by_the_model_and_ids_are_not(self):
        """`phonemes=` takes both spellings a caller actually holds: the STRING every G2P returns, and
        ids already encoded. It used to take only the second -- `[int(p) for p in phonemes]`, byte for
        byte the `tokens=` branch -- so the string form died on `invalid literal for int(): 'h'` and the
        two parameters were one parameter under two names.

        Which one is passed decides more than convenience: encoding here is what applies the model's own
        BOS/EOS assembly, and that assembly is precisely what a bring-your-own-G2P caller cannot know
        about (Kokoro's driver needs its ids wrapped with 0 at both ends). `tokens=` deliberately keeps
        going through untouched, for the caller who has already done it.
        """
        handle = _FakeHandle([[0.0]], contract=dict(_TTS_PHONEME_CONTRACT, text_frontend="phonemes"))
        _model(handle).text2speech.infer(phonemes="hɛ")
        assert handle.calls[0]["tokens"] == [10.0, 11.0, 12.0], "the table encoded it"

        handle = _FakeHandle([[0.0]], contract=dict(_TTS_PHONEME_CONTRACT, text_frontend="phonemes"))
        _model(handle).text2speech.infer(phonemes=[7, 8])
        assert handle.calls[0]["tokens"] == [7.0, 8.0], "ids pass through, assembly included"

    def test_a_phoneme_string_with_no_table_says_the_table_is_missing(self):
        """The str form needs the embedded table; the id form does not. A model exported before the
        table was written must say so rather than failing on an int() of a letter."""
        handle = _FakeHandle([], contract=dict(_TTS_PHONEME_CONTRACT, text_frontend="phonemes"),
                             vocab=None)
        with pytest.raises(loom.UnsupportedTask) as excinfo:
            _model(handle).text2speech.infer(phonemes="hɛ")
        assert "re-export" in str(excinfo.value)

    def test_a_phoneme_model_with_no_table_says_the_table_is_missing(self):
        """A model exported before the symbol table was written. The fix is a re-export, not a
        different call, and the message says which."""
        handle = _FakeHandle([], contract=dict(_TTS_PHONEME_CONTRACT, text_frontend="phonemes"),
                             vocab=None)
        with pytest.raises(loom.UnsupportedTask) as excinfo:
            _model(handle).text2speech.infer("hello")
        assert "re-export" in str(excinfo.value)

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


class TestSampleRateFallback:
    """A declared rate wins; a caller-supplied one fills the gap; both beat guessing silently.

    Only Supertonic declares its rate today, and the other four TTS families run at 22.05, 24 and
    44.1 kHz -- so the fallback is usually wrong, and audio at the wrong rate does not fail, it plays at
    the wrong speed. The warning is the only signal a caller gets that the number was invented.
    """

    def test_a_declared_rate_wins_and_says_nothing(self):
        """The export read it off the checkpoint; the caller is guessing. So an argument does NOT
        override a declaration -- it fills in for its absence."""
        handle = _FakeHandle([[0.0]], contract=_TTS_PHONEME_CONTRACT)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            audio = _model(handle).text2speech.infer(phonemes=[1], sample_rate=8000)
        assert audio.sample_rate == 22050
        assert not caught

    def test_the_argument_is_used_when_nothing_is_declared(self):
        handle = _FakeHandle([[0.0]], contract=dict(_TTS_PHONEME_CONTRACT, sample_rate=0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            audio = _model(handle).text2speech.infer(phonemes=[1], sample_rate=24000)
        assert audio.sample_rate == 24000
        assert len(caught) == 1, "still a guess, so still worth saying"
        assert "24000 Hz is used" in str(caught[0].message)

    def test_the_default_is_16000_and_comes_from_the_signature(self):
        handle = _FakeHandle([[0.0]], contract=dict(_TTS_PHONEME_CONTRACT, sample_rate=0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            audio = _model(handle).text2speech.infer(phonemes=[1])
        assert audio.sample_rate == 16000
        assert issubclass(caught[0].category, RuntimeWarning)
        # The message has to say what goes wrong if the guess is wrong, not merely that it guessed.
        assert "wrong speed" in str(caught[0].message)

    def test_the_default_is_visible_where_a_caller_looks_for_it(self):
        """In the signature rather than a module constant, so `help(...)` shows it and a caller can
        replace it per call."""
        import inspect

        sig = inspect.signature(loom.Text2Speech._infer)
        assert sig.parameters["sample_rate"].default == 16000
