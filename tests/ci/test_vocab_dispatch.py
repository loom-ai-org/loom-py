"""What the binding does with a GGUF whose vocabulary it has never seen.

Still CI, still no checkpoint: every file here is a few hundred bytes written by `gguf` in a tmpdir --
a `tokenizer.ggml.model` tag, the one KV `GgufModel::load` insists on, and a placeholder tensor. What
they exercise is the one decision a fake handle cannot reach, because the C++ makes it during
construction: `Tokenizer::load`'s dispatch on that tag.

The regression these exist for: that dispatch used to end in an `else` that handed every unrecognized
tag to `Vocab::load`, which THROWS on anything but "llama"/"t5" -- and it threw inside `Model`'s
constructor, so a GGUF carrying a vocabulary the binding had not been taught was not a model missing a
tokenizer, it was a model that would not load. Teaching the exporter to write Supertonic's grapheme
table into its GGUF would have bricked every Supertonic model, which is how this was found.
"""
import numpy as np
import pytest
from gguf import GGUFWriter

import loom

# GgufModel::load requires this KV even when there is no graph to run.
EMPTY_TOPOLOGY = '{"version": 1, "nodes": []}'


def _write_gguf(path, tokenizer_model=None, extra=None):
    w = GGUFWriter(str(path), "loom-vocab-dispatch-fixture")
    w.add_string("loom.architecture", "vocab_dispatch_test")
    w.add_string("model.graph_topology", EMPTY_TOPOLOGY)
    if tokenizer_model is not None:
        w.add_tokenizer_model(tokenizer_model)
    for key, value in (extra or {}).items():
        if isinstance(value, str):
            w.add_string(key, value)
        else:
            w.add_array(key, value)
    # A GGUF with zero tensors hits a ggml_backend edge case when the buffer is sized; nothing reads it.
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def _supertonic_table():
    """Printable ASCII -> 0.., in the same flat BMP-sized shape as the real unicode_indexer.json."""
    table = [-1] * 65536
    for cp in range(32, 127):
        table[cp] = cp - 32
    return table


class TestUnknownVocabularyDoesNotBreakLoading:
    def test_an_unrecognized_tokenizer_tag_leaves_the_model_loadable(self, tmp_path):
        path = _write_gguf(tmp_path / "unknown.gguf", tokenizer_model="some-future-family")
        model = loom.Model.from_file(path)  # the regression: this used to raise
        assert model.tokenizer is None

    def test_a_recognized_tag_with_no_vocabulary_data_is_loud_rather_than_silent(self, tmp_path):
        """The other half of the same decision, and it goes the other way on purpose. An UNKNOWN tag is
        a valid file this binding has not been taught, so it degrades to "no tokenizer". A tag it DOES
        know, with the data missing, is a malformed file -- silently reporting no tokenizer there would
        turn a broken export into a mystery at the call site instead of an error at the load."""
        path = _write_gguf(tmp_path / "empty_bert.gguf", tokenizer_model="bert")
        with pytest.raises(loom.LoomError, match="tokenizer.ggml.tokens"):
            loom.Model.from_file(path)

    def test_no_tokenizer_kv_at_all_is_still_the_no_vocabulary_answer(self, tmp_path):
        path = _write_gguf(tmp_path / "none.gguf")
        model = loom.Model.from_file(path)
        assert model.tokenizer is None
        with pytest.raises(RuntimeError, match="no tokenizer vocabulary"):
            model.tokenize("hello")


class TestSupertonicGraphemeVocabulary:
    """The family the dispatch fix was needed for: a TTS model that encodes text itself."""

    @pytest.fixture
    def model(self, tmp_path):
        path = _write_gguf(
            tmp_path / "supertonic.gguf", tokenizer_model="supertonic",
            extra={"tokenizer.ggml.supertonic.codepoint_to_id": _supertonic_table(),
                    "tokenizer.ggml.supertonic.default_lang": "es"},
        )
        return loom.Model.from_file(path)

    def test_it_reads_back_as_a_tokenizer_like_any_other_family(self, model):
        assert model.tokenizer is not None
        assert model.tokenizer.kind == "supertonic"
        # The VOCABULARY (95 printable ASCII ids), not the 65536-entry lookup table. Reporting the
        # table's length here would be off by three orders of magnitude.
        assert model.tokenizer.size == 95

    def test_an_omitted_lang_uses_the_file_s_own_default(self, model):
        expected = [ord(c) - 32 for c in "<es>hi.</es>"]
        assert model.tokenize("hi") == expected
        assert model.tokenize("hi", lang="es") == expected

    def test_a_named_lang_wins(self, model):
        assert model.tokenize("hi", lang="ko") == [ord(c) - 32 for c in "<ko>hi.</ko>"]

    def test_ids_round_trip_back_to_the_preprocessed_text(self, model):
        # detokenize inverts the codepoint table, not the preprocessing -- so the wrap and the
        # inserted period come back too, and asserting the whole string is what says so.
        assert model.detokenize(model.tokenize("hi")) == "<es>hi.</es>"

    def test_floats_detokenize_because_floats_are_what_infer_returns(self, model):
        assert model.detokenize([float(ord(c) - 32) for c in "hi"]) == "hi"


    def test_an_absent_default_lang_kv_falls_back_to_en(self, tmp_path):
        """Every Supertonic GGUF written before that KV existed has none, and "en" is the real
        `tokenize_str`'s own default -- so an old file tokenizes the same way it always did."""
        path = _write_gguf(
            tmp_path / "no_lang.gguf", tokenizer_model="supertonic",
            extra={"tokenizer.ggml.supertonic.codepoint_to_id": _supertonic_table()},
        )
        model = loom.Model.from_file(path)
        assert model.tokenizer.default_lang == "en"
        assert model.tokenize("hi") == [ord(c) - 32 for c in "<en>hi.</en>"]


class TestLanguageArgumentIsRejectedWhereItCannotBeHonoured:
    """A dropped argument is the failure this rejection prevents: a caller asking for Korean and
    quietly getting whatever the tokenizer does by default."""

    @pytest.fixture
    def byte_model(self, tmp_path):
        # A minimal real ByT5-family vocabulary: pad/eos/unk then one id per byte.
        path = str(tmp_path / "byt5.gguf")
        w = GGUFWriter(path, "loom-vocab-dispatch-fixture")
        w.add_string("loom.architecture", "vocab_dispatch_test")
        w.add_string("model.graph_topology", EMPTY_TOPOLOGY)
        w.add_tokenizer_model("byt5")
        w.add_token_list(["<pad>", "</s>", "<unk>"] + [""] * 256)
        w.add_pad_token_id(0)
        w.add_eos_token_id(1)
        w.add_unk_token_id(2)
        w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return loom.Model.from_file(path)

    def test_a_vocabulary_with_no_language_concept_refuses_a_lang(self, byte_model):
        assert byte_model.tokenizer.kind == "byt5"
        with pytest.raises(loom.LoomError, match="takes no language argument"):
            byte_model.tokenize("hi", lang="ko")

    def test_the_same_vocabulary_tokenizes_fine_without_one(self, byte_model):
        assert byte_model.tokenize("hi") == [ord("h") + 3, ord("i") + 3, 1]
