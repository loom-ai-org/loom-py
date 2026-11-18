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


# -- the `kokoro` style (Paradee, loom.cpp ADR-135) --------------------------------------------------

def test_kokoro_without_its_front_ends_runs_between_marks_and_keeps_them(monkeypatch):
    monkeypatch.setitem(phonemizers._KOKORO_NATIVE, "a", None)
    calls = []

    def g2p(text, lang):
        calls.append(text.strip())
        return "ˈtɛst"
    monkeypatch.setitem(phonemizers._PROVIDERS, "ipa", g2p)
    out = phonemizers.phonemize_styled("Hello world, this is it. Bye!", "kokoro", language="en-US")
    # The default G2P drops punctuation, so it runs between the marks; each mark follows its word.
    assert calls == ["Hello world", "this is it", "Bye"]
    assert out == "tˈɛst, tˈɛst. tˈɛst!"


def test_kokoro_with_its_front_end_installed_is_that_front_end(monkeypatch):
    monkeypatch.setitem(phonemizers._KOKORO_NATIVE, "b", lambda text: f"<misaki {text}>")
    monkeypatch.setitem(phonemizers._KOKORO_NATIVE, "fr", lambda text: f"<paradee {text}>")
    assert phonemizers.phonemize_styled("Hi.", "kokoro", language="en-GB") == "<misaki Hi.>"
    assert phonemizers.phonemize_styled("Salut.", "kokoro", language="fr-FR") == "<paradee Salut.>"


@pytest.mark.parametrize("language,ipa,folded", [
    # espeak's Hindi spells the palatal stops ɟ and c and the breathy glottal h, with no dental mark.
    ("hi", "aːdʒ bəɦʊt̪ ətʃtʃʰaː", "aːɟ bəhʊt əccʰaː"),
    # Stress before the vowel, as espeak writes it.
    ("es", "la βiβljoˈteka ˈnweβe", "la βiβljotˈeka nwˈeβe"),
    # Ties espeak makes one symbol (misaki's EspeakG2P compression).
    ("fr", "dʒinz tʃɛk", "ʤinz ʧɛk"),
    # English is misaki's.
    ("en-US", "ˈtaɪmz", "tˈImz"),
])
def test_the_kokoro_fold_per_language(language, ipa, folded):
    assert phonemizers.fold(ipa, "kokoro", language=language) == folded
