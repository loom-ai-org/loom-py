"""The Python layer, without a model.

Everything here runs in seconds against the built extension and no GGUF, which is what makes it CI:
the questions are about argument coercion and error messages, and those are exactly the things that
are wrong in a way nobody notices until a wrong number reaches a model.

The half that needs a real model is `tests/gate/`.
"""
import builtins
import sys
import types

import pytest

import loom


class TestPackage:
    def test_it_imports_and_exports_what_it_says(self):
        assert loom.__all__ == ["Model", "LoomError", "download", "__version__"]
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
