"""Voice files (loom.cpp ADR-045): `model.voices`, `model.voice(...)` and `text2speech.infer(voice=...)`.

A voice file is a small GGUF whose tensors are driver inputs by name, stamped with the fingerprint of
the weights it was made for; the model declares the same fingerprint, and the ENGINE refuses a
mismatch (`loom::load_voice`). What is pinned here is the door: how a name resolves (beside the model
file, then the model's Hub repo), that the loaded inputs reach the driver and an explicit one still
wins, and that a mismatched file fails loudly rather than producing a different model's audio.
"""
import numpy as np
import pytest
from gguf import GGUFWriter

import loom

COMPAT = "ab" * 16
# A driver that returns the `voice_kv` it was handed, or [-1] when it was handed none -- so the test
# reads back exactly what the door passed.
DRIVER = """
function infer(inputs)
    return inputs.voice_kv or {-1}
end
"""


def _model(path, compat=COMPAT, voices=("builtin",)):
    w = GGUFWriter(str(path), "voice-door-test")
    w.add_string("loom.architecture", "voice_door_test")
    w.add_string("model.graph_topology", '{"version": 1, "nodes": []}')
    w.add_string("model.driver_script", DRIVER)
    w.add_string("loom.task", "text-to-speech")
    w.add_string("loom.input.kind", "text")
    w.add_string("loom.output.kind", "audio")
    w.add_uint32("loom.sample_rate", 24000)
    if compat is not None:
        w.add_string("loom.voice.compat", compat)
    if voices:
        w.add_array("loom.tts.voices", list(voices))
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return loom.Model.from_file(path)


def _voice(path, values, compat=COMPAT, arch="voice_door_test", name="v"):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(path), "loom-voice")
    w.add_string("loom.voice.architecture", arch)
    w.add_string("loom.voice.compat", compat)
    w.add_string("loom.voice.name", name)
    w.add_string("loom.voice.license", "CC0-1.0")
    w.add_string("loom.voice.origin", "test")
    w.add_tensor("voice_kv", np.asarray(values, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture
def model(tmp_path):
    m = _model(tmp_path / "model.gguf")
    _voice(tmp_path / "voices" / "marius.gguf", [1.0, 2.0, 3.0], name="marius")
    _voice(tmp_path / "voices" / "stranger.gguf", [9.0], compat="cd" * 16, name="stranger")
    return m


def test_voices_lists_the_builtin_first_then_the_files_beside_the_model(model):
    assert model.voices == ["builtin", "marius", "stranger"]


def test_a_name_resolves_beside_the_model_and_its_inputs_reach_the_driver(model):
    voice = model.voice("marius")
    assert voice.name == "marius" and voice.license == "CC0-1.0"
    assert list(voice.inputs["voice_kv"]) == [1.0, 2.0, 3.0]
    assert model.text2speech.infer(tokens=[5], voice="marius").samples == [1.0, 2.0, 3.0]


def test_a_path_and_a_loaded_voice_work_too(model, tmp_path):
    path = _voice(tmp_path / "elsewhere" / "mine.gguf", [4.0, 5.0])
    assert model.text2speech.infer(tokens=[5], voice=path).samples == [4.0, 5.0]
    loaded = model.voice(path)
    assert model.text2speech.infer(tokens=[5], voice=loaded).samples == [4.0, 5.0]


def test_no_voice_is_the_models_own_default(model):
    assert model.text2speech.infer(tokens=[5]).samples == [-1.0]


def test_an_explicit_driver_input_still_wins_over_the_voice(model):
    assert model.text2speech.infer(tokens=[5], voice="marius", voice_kv=[7.0]).samples == [7.0]


def test_a_voice_for_other_weights_is_refused_by_the_engine(model):
    with pytest.raises(loom.LoomError, match="other weights"):
        model.voice("stranger")


def test_a_voice_for_another_architecture_is_refused(model, tmp_path):
    path = _voice(tmp_path / "other_arch.gguf", [1.0], arch="another_model")
    with pytest.raises(loom.LoomError, match="another_model"):
        model.voice(path)


def test_an_unknown_name_says_what_is_available(model):
    with pytest.raises(FileNotFoundError, match="marius"):
        model.voice("nobody")


def test_a_model_without_a_fingerprint_takes_no_voice_files(tmp_path):
    m = _model(tmp_path / "plain.gguf", compat=None, voices=())
    assert m.voices == []
    with pytest.raises(ValueError, match="takes no voice files"):
        m.voice("anything")


def test_a_name_not_on_disk_is_fetched_from_the_models_hub_repo(model, tmp_path, monkeypatch):
    """`from_pretrained` records the repo; a voice named later comes from its `voices/`, downloaded
    beside the model the way `hf_hub_download` lays a snapshot out."""
    import sys
    import types

    remote = _voice(tmp_path / "remote_src" / "remote.gguf", [6.0, 6.0], name="remote")
    fetched = []

    hub = types.ModuleType("huggingface_hub")
    hub.list_repo_files = lambda repo_id, revision=None, token=None: ["model.gguf", "voices/remote.gguf"]

    def hf_hub_download(repo_id, filename, revision=None, cache_dir=None, token=None):
        fetched.append((repo_id, filename, revision))
        return str(remote)

    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    model._hub = dict(repo_id="loom-ai-org/x", revision="abc", cache_dir=None, token=None)
    assert "remote" in model.voices
    assert model.text2speech.infer(tokens=[5], voice="remote").samples == [6.0, 6.0]
    assert fetched == [("loom-ai-org/x", "voices/remote.gguf", "abc")]
