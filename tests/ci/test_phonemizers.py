"""The `dizione` phoneme style (tiny-voca, loom.cpp ADR-133): the reference's sentence cut, and the fold
that stands in for dizione when it is not installed. dizione itself is not needed here."""
import pytest

from loom import phonemizers


def test_sentences_are_cut_as_speak_text_cuts_them():
    assert phonemizers.dizione_sentences("Hello! This is tiny voca. Dr. Smith said hi") == [
        "Hello!", "This is tiny voca.", "Dr. Smith said hi."]
    # A mark before a closing quote still ends the sentence; an initial does not.
    assert phonemizers.dizione_sentences('She said "go!" Then J. Doe left.') == ['She said "go!"', "Then J. Doe left."]
    # A face after the last word is no mark, so the text gets its full stop there (upstream's answer).
    assert phonemizers.dizione_sentences("Fine. :-) ") == ["Fine. :-)."]


def test_the_fold_writes_dizione_symbols_one_sentence_wrapped():
    folded = phonemizers._fold_dizione("həˈloʊ ˈwɝld! ˈbɑtəl ˈkɑɹ", "en-us")
    assert folded == "<bos> h ə l ˈoʊ <sp> w ˈɜː l d ! b ˈɑː t əl <sp> k ˈɑːɹ <eos>"


def test_a_syllabic_l_is_taken_only_before_no_vowel():
    assert " ə l " in phonemizers._fold_dizione("həˈloʊ", "en-us")
    assert phonemizers._fold_dizione("ˈbɑtəl", "en-us").endswith("t əl <eos>")


def test_without_dizione_the_door_folds_the_registered_g2p(monkeypatch):
    monkeypatch.setattr(phonemizers, "_dizione_native", lambda: None)
    monkeypatch.setitem(phonemizers._PROVIDERS, "ipa", lambda text, lang: "həˈloʊ")
    out = phonemizers.phonemize_styled("Hello there. Bye", "dizione", language="en-us")
    # Two sentences, each its own utterance.
    assert out.count("<bos>") == 2 and out.count("<eos>") == 2


def test_with_dizione_the_door_is_dizione_per_sentence(monkeypatch):
    class Fake:
        @staticmethod
        def phonemize(sentence):
            return f"<bos> {sentence} <eos>"
    monkeypatch.setattr(phonemizers, "_dizione_native", lambda: Fake)
    assert phonemizers.phonemize_styled("Hi. Bye", "dizione") == "<bos> Hi. <eos> <bos> Bye. <eos>"


def test_an_unknown_style_is_still_refused():
    with pytest.raises(LookupError):
        phonemizers.fold("a", "nonesuch")
