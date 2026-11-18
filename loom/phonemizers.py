"""Grapheme-to-phoneme: the one step of a text-to-speech pipeline that is not in the GGUF.

WHY IT IS OUT HERE. Everything else a TTS model needs travels with it -- the graphs, the driver, and now
the phoneme symbol table its checkpoint always carried. G2P does not, because it is a property of the
LANGUAGE rather than of any checkpoint: the rules that turn "hello" into `həˈloʊ` are the same rules
whichever model is about to say it, and baking them into files would mean re-exporting every model to
fix a pronunciation (loom.cpp docs/HIGH-LEVEL-API.md §2/§5).

WHY IT IS OPTIONAL. This package declares no runtime dependencies, deliberately -- loading a GGUF and
running its driver needs none, and a caller who only runs an LM or an ASR model should not acquire numpy
and a language-data package to do it. So the phonemizer is an extra:

    pip install "loom-py-rt[phonemes]"

Without it, `synthesize(phonemes=...)` and `infer` work exactly as before and only the text door is
absent, with an error that says which install fixes it.

WHAT REPLACES THIS LATER. `orthography2ipa` is Apache-2.0 rule-based transduction -- ~900 language JSON
specs plus one language-agnostic engine, no weights -- and the plan is a C++ port of it in the engine
(BACKLOG.md Task #79). When that lands, the engine becomes the provider and this Python path is RETIRED
rather than kept beside it: two implementations of one conversion, selected by whether an extra happens
to be installed, is the same defect the LM decode loop had in two hosts -- the same text yielding
different audio in two environments. The registry below survives that transition unchanged; it stops
being how phonemization is provided and remains how a caller substitutes their own.
"""
from __future__ import annotations

import os
import re
import warnings
from typing import Callable, Dict, Optional

#: `{alphabet: callable(text, language) -> str}`. Keyed by alphabet rather than by package, because what
#: a model declares in `loom.text.phoneme_alphabet` is the alphabet it was trained on -- which provider
#: produced it is not something the model has an opinion about.
_PROVIDERS: Dict[str, Callable[[str, str], str]] = {}

#: `{resolved language: source}` as named to `set_lexicon`, newest wins -- what `lexicons()` reports.
_LEXICONS: Dict[str, str] = {}

#: `{resolved language: orthography2ipa.G2P}`, each built once with `Pipeline(lexicon=...)` over that
#: language's entries. An engine of its own rather than the library's process-wide registry
#: (`register_lexicon`, deprecated in orthography2ipa 7.123 and removed in 8.0): the lexicon then reaches
#: loom's transcriptions and nothing else in the process, and building the engine once keeps the spec
#: load off every call.
_ENGINES: Dict[str, object] = {}


def register(alphabet: str, phonemize: Callable[[str, str], str]) -> None:
    """Register a G2P for an alphabet, replacing any previous one.

    `phonemize(text, language) -> str` returns the phoneme string the model's own table will be asked to
    encode. Symbols outside that table are dropped by the vocabulary rather than refused, because a
    rule-based engine emits a superset of what any one checkpoint was trained on.
    """
    _PROVIDERS[alphabet] = phonemize


def set_lexicon(source: Optional[str | os.PathLike] = None, *, language: str = "en") -> None:
    """Point the default provider at a pronunciation lexicon for `language`, or `None` to clear one.

    `source` is a `word<TAB>ipa` TSV -- a local path, an `http(s)://` URL, or a Hugging Face
    `hf://<repo>/<path>` id, all three resolved by orthography2ipa's `load_lexicon`. It is read here,
    once, into the engine loom transcribes `language` with, so that a file which yields nothing is
    reported at the line that named it -- see WHY IT IS VERIFIED HERE below. The lexicon is loom's:
    other users of orthography2ipa in the process do not see it.

    **WHY THIS EXISTS.** orthography2ipa transduces by RULE, and English is one of the deep-orthography
    languages its own documentation names as unreachable that way -- "time" comes out `tɪm` rather than
    `tˈaɪm`, "friend" as `fɹiːnd`, and stress is absent entirely because the `en-GB` spec declares no
    stress rules at all. None of that is a search-quality problem and no parameter fixes it: `search=
    "beam"` returns the greedy string unchanged at every width tried, because there are no competing
    candidates to reorder. A lexicon is the mechanism the library provides for exactly this, entries may
    carry stress marks, and with one the same sentence comes back matching espeak's output.

    An unlisted word still falls back to the rules, so coverage is what determines quality; this is an
    overlay, not a replacement.

    `language` is resolved the way orthography2ipa resolves it, so `"en"` reaches the `en-GB` spec and a
    lexicon set for either is found when either is synthesised. Keyed by the caller's string instead, a
    lexicon set for `"en"` would silently never apply to `"en-GB"`.

    **WHY IT IS VERIFIED HERE.** The parse skips, without complaint, every line that is not exactly
    `word<TAB>ipa` -- so a file of the wrong SHAPE loads as an empty overlay, and an empty overlay is
    byte-identical to no overlay at all: registration succeeds, transcription falls back to the rules,
    and the audio comes back unstressed with nothing anywhere saying why. That is not hypothetical.
    Running the documented `sed` over the GitHub *blob* page for ipa-dict instead of the raw file
    yields 700 lines of HTML, zero of them entries, and the only symptom is that `set_lexicon` appears
    not to be taken into account. So a zero-entry lexicon warns. Reading it here is not extra work: the
    engine needs the entries before its first transcription anyway.

    Raises `LookupError` when orthography2ipa is not installed, or is older than 7.123 (no per-engine
    lexicon), because a lexicon set on a provider that does not exist would otherwise be accepted and
    never applied. A source that cannot be READ raises whatever reading it raises -- `FileNotFoundError`
    for a path that is not there, a fetch error for a URL or an `hf://` id -- and replaces nothing: the
    previous lexicon for that language stays.
    """
    try:
        import orthography2ipa
    except ImportError:
        raise LookupError(
            "a lexicon configures the default phonemizer, and orthography2ipa is not installed. "
            "Install it with `pip install \"loom-py-rt[phonemes]\"`. A provider registered with "
            "`loom.phonemizers.register(...)` brings its own pronunciations and needs none of this."
        ) from None

    if not hasattr(orthography2ipa, "Pipeline"):
        raise LookupError(
            f"a lexicon needs orthography2ipa 7.123 or newer, and "
            f"{getattr(orthography2ipa, '__version__', 'an older one')} is installed. "
            "Upgrade it with `pip install -U \"loom-py-rt[phonemes]\"`."
        )
    code = orthography2ipa.resolve(language)
    if source is None:
        _LEXICONS.pop(code, None)
        _ENGINES.pop(code, None)
        return
    entries = orthography2ipa.load_lexicon(source)      # raises before anything is replaced
    _LEXICONS[code] = str(source)
    _ENGINES[code] = orthography2ipa.G2P(code, pipeline=orthography2ipa.Pipeline(lexicon=entries))
    if not entries:
        warnings.warn(
            f"the lexicon at {source} holds no entries, so it will change nothing -- an empty overlay "
            f"behaves exactly like no lexicon at all. Every line must be `word<TAB>ipa`; anything else "
            f"is skipped without complaint, so a file of the wrong shape parses to nothing. "
            f"`orthography2ipa.validate_lexicon_text(text)` names the lines it rejected.",
            RuntimeWarning, stacklevel=2,
        )


def lexicons() -> Dict[str, str]:
    """`{resolved language: source}` for every lexicon set here. A copy; `set_lexicon` is the door."""
    return dict(_LEXICONS)


def available(alphabet: str = "ipa") -> bool:
    """Whether anything can phonemize into `alphabet` right now."""
    return alphabet in _PROVIDERS or _load_default(alphabet) is not None


def phonemize(text: str, *, alphabet: str = "ipa", language: str = "en") -> str:
    """Text to phoneme symbols, through the registered provider for `alphabet`."""
    provider = _PROVIDERS.get(alphabet) or _load_default(alphabet)
    if provider is None:
        raise LookupError(
            f"no phonemizer registered for {alphabet!r}, and orthography2ipa is not installed. "
            f"Install it with `pip install \"loom-py-rt[phonemes]\"`, or register your own with "
            f"`loom.phonemizers.register({alphabet!r}, fn)`. Passing `phonemes=` directly needs neither."
        )
    return provider(text, language)



# -- phoneme styles --------------------------------------------------------------------------------
#
# A phoneme TABLE says which symbols a checkpoint knows; it does not say which CONVENTIONS its training
# data wrote them in. A rule-based G2P plus a dictionary lexicon (ipa-dict) writes stress before the
# syllable's onset (`ˈkwɪk`), stresses every monosyllable (`ˈðə`), and leaves English vowels unmarked for
# length (`i`, `ɑ`, `ɔ`). espeak -- what every Piper voice and every voice distilled from one was trained
# on -- writes `kwˈɪk`, `ðə`, `iː`, `ɑː`, `ɔː`. A large model hears through that (Piper's VITS is
# word-perfect either way); a 1.46M-parameter student does not: sanoTTS amy went from 66.8% WER to 11.1%
# on 30 LibriSpeech sentences when its input was folded to espeak's conventions, against 8.5% for
# upstream's own espeak-style G2P (loom.cpp ADR-071).
#
# So a model DECLARES the style it was trained on (`loom.tts.phoneme_style`) and the text door folds the
# G2P's output to it before encoding. Only text the door phonemized is folded: `phonemes=` from the
# caller is taken as given, because a caller who brings their own G2P has already chosen conventions.

_STRESS = "ˈˌ"
_VOWELS = frozenset("aeiouæɑɒɔəɛɜɝɚɪʊʌɐᵻᵊAIWOY")


def _stress_before_vowel(word: str) -> str:
    """Move each stress mark from a syllable's onset to its vowel: `ˈkwɪk` -> `kwˈɪk`."""
    out, pending = [], ""
    for ch in word:
        if ch in _STRESS:
            pending = ch
            continue
        if pending and ch in _VOWELS:
            out.append(pending)
            pending = ""
        out.append(ch)
    if pending:
        out.append(pending)
    return "".join(out)


def _replace_all(word: str, pairs) -> str:
    for needle, replacement in pairs:
        word = word.replace(needle, replacement)
    return word


# misaki's compressed symbols back to espeak's, and the two IPA spellings espeak never uses.
_TO_ESPEAK = (("A", "eɪ"), ("I", "aɪ"), ("W", "aʊ"), ("O", "oʊ"), ("Y", "ɔɪ"), ("ʤ", "dʒ"), ("ʧ", "tʃ"),
              ("ᵊl", "əl"), ("ᵊ", "ə"), ("T", "ɾ"), ("ɜɹ", "ɜː"), ("ʰ", ""), ("ɫ", "l"), ("r", "ɹ"))


def _fold_espeak(phonemes: str, language: str) -> str:
    english = language.lower().startswith("en")
    words = []
    for word in phonemes.split():
        word = _replace_all(_stress_before_vowel(word), _TO_ESPEAK)
        if english:
            # espeak en-us: a stressed r-coloured vowel is `ɜː`, an unstressed one `ɚ`; a stressed
            # schwa is `ʌ`; i/u/ɑ/ɔ/ɜ are long, except a word-final unstressed `i`.
            word = re.sub(r"([ˈˌ])ɝ", r"\1ɜː", word).replace("ɝ", "ɚ")
            word = re.sub(r"([ˈˌ])əɹ", r"\1ɜː", word).replace("əɹ", "ɚ")
            word = re.sub(r"([ˈˌ])ə", r"\1ʌ", word)
            word = word.replace("ɒ", "ɑ")
            word = re.sub(r"ɔ(?![ːɪ])", "ɔː", word)
            word = re.sub(r"ɑ(?!ː)", "ɑː", word)
            word = re.sub(r"ɜ(?!ː)", "ɜː", word)
            word = re.sub(r"u(?!ː)", "uː", word)
            word = re.sub(r"i(?![ːə])", "iː", word)
            # Shortened only when the word is stressed elsewhere: the mark sits right before the vowel
            # it stresses, so a final `iː` it does not precede is unstressed. An unmarked word (`wiː`)
            # keeps its length, as espeak gives it.
            if word.endswith("iː") and any(c in word for c in _STRESS) and word[-3:-2] not in _STRESS:
                word = word[:-1]
        words.append(word)
    return " ".join(words)


# espeak/IPA spellings to misaki's (Kokoro's alphabet), after misaki's own `EspeakFallback.E2M`: the
# diphthongs and affricates become one symbol, length is dropped, a bare `e` is `A`.
_TO_MISAKI = (("aɪ", "I"), ("aʊ", "W"), ("dʒ", "ʤ"), ("eɪ", "A"), ("tʃ", "ʧ"), ("ɔɪ", "Y"), ("oʊ", "O"),
              ("əʊ", "O"), ("ɚ", "əɹ"), ("ɝ", "ɜɹ"), ("ɜːɹ", "ɜɹ"), ("ɜː", "ɜɹ"), ("r", "ɹ"), ("ɐ", "ə"),
              ("ɫ", "l"), ("ʰ", ""), ("ɒ", "ɑ"), ("ɪə", "iə"), ("ː", ""), ("e", "A"), ("o", "ɔ"),
              ("ɾ", "T"))


def _fold_misaki(phonemes: str, language: str) -> str:
    return " ".join(_replace_all(_stress_before_vowel(w), _TO_MISAKI) for w in phonemes.split())


# IPA to ARPAbet with stress digits, the alphabet g2p_en writes and CMUdict is in (AuroraTTS-100K's).
# Unlike the two folds above this changes ALPHABET as well as conventions, so its output is a token
# string the model's table spells: phonemes space-separated, words joined by `|` -- "DH AH0 | K W IH1 K".
# The engine skips the separating spaces (no symbol holds one); with the boundary spelled as a space too,
# `D HH` (adhere) and `DH` would read the same.
#
# It reads every spelling the door's lexicons and rules produce: ipa-dict's (stress before the onset,
# `ɫ`, `ɝ`), misaki's compressed symbols, orthography2ipa's, espeak's length marks, and the CMUdict
# lexicon the AuroraTTS card ships, which is written from CMUdict so that this fold returns every one of
# its 135,166 entries exactly (it marks a syllable break with `.` wherever two phones would otherwise
# spell a third: `AO IH0` against `OY`, `AH0 R` against `ER0`, `T SH` against `CH`). A vowel takes the
# stress of the mark before it (1 for `ˈ`, 2 for `ˌ`) and 0 without one; a schwa is AH0 whatever marks
# it, which is CMUdict's reading of orthography2ipa's stressed monosyllables (`ˈðə`, "the" = DH AH0).
_ARPA_VOWELS = (("ɜːɹ", "ER"), ("ɜɹ", "ER"), ("əɹ", "ER0"), ("ɪə", "IH R"), ("ɛə", "EH R"), ("ʊə", "UH R"),
                ("aɪ", "AY"), ("aʊ", "AW"), ("eɪ", "EY"), ("oʊ", "OW"), ("əʊ", "OW"), ("ɔɪ", "OY"),
                ("ɜː", "ER"), ("iː", "IY"), ("uː", "UW"), ("ɑː", "AA"), ("ɔː", "AO"), ("ɝ", "ER"),
                ("ɚ", "ER0"), ("ɜ", "ER"), ("i", "IY"), ("u", "UW"), ("ɑ", "AA"), ("ɒ", "AA"), ("ɔ", "AO"),
                ("æ", "AE"), ("ɛ", "EH"), ("e", "EY"), ("ɪ", "IH"), ("ᵻ", "IH"), ("ʊ", "UH"), ("ʌ", "AH"),
                ("ə", "AH0"), ("ɐ", "AH0"), ("ᵊ", "AH0"), ("a", "AA"), ("o", "OW"),
                ("A", "EY"), ("I", "AY"), ("W", "AW"), ("O", "OW"), ("Y", "OY"))
_ARPA_CONSONANTS = (("tʃ", "CH"), ("dʒ", "JH"), ("ʧ", "CH"), ("ʤ", "JH"), ("θ", "TH"), ("ð", "DH"),
                    ("ʃ", "SH"), ("ʒ", "ZH"), ("ŋ", "NG"), ("ɹ", "R"), ("r", "R"), ("ɾ", "T"), ("T", "T"),
                    ("ɫ", "L"), ("l", "L"), ("j", "Y"), ("ɡ", "G"), ("g", "G"), ("h", "HH"), ("b", "B"),
                    ("d", "D"), ("f", "F"), ("k", "K"), ("m", "M"), ("n", "N"), ("p", "P"), ("s", "S"),
                    ("t", "T"), ("v", "V"), ("w", "W"), ("z", "Z"))
_ARPA_TABLE = sorted([(k, v, True) for k, v in _ARPA_VOWELS] + [(k, v, False) for k, v in _ARPA_CONSONANTS],
                     key=lambda entry: -len(entry[0]))
_ARPA_VOWEL_STARTS = frozenset(k[0] for k, _ in _ARPA_VOWELS)
_SYLLABIC = "\u0329"


def _arpabet_word(word: str) -> list:
    phones, stress, i = [], "0", 0
    while i < len(word):
        ch = word[i]
        if ch == ".":                    # a syllable break: no phone, and no digraph across it
            i += 1
            continue
        if ch in _STRESS:
            stress = "1" if ch == "ˈ" else "2"
            i += 1
            continue
        if ch == _SYLLABIC:              # n̩, l̩: CMUdict spells a syllabic consonant AH0 + consonant
            if phones and not phones[-1][-1].isdigit():
                phones.insert(len(phones) - 1, "AH0")
            i += 1
            continue
        for key, arpa, vowel in _ARPA_TABLE:
            if not word.startswith(key, i):
                continue
            j = i + len(key)
            # An r-coloured vowel's `ɹ` before another vowel is that syllable's onset ("arabia",
            # `əɹˈeɪbiə` = AH0 R EY1 ...), so the vowel is matched without it.
            if vowel and key.endswith("ɹ"):
                k = j
                while k < len(word) and word[k] in _STRESS:
                    k += 1
                if k < len(word) and word[k] in _ARPA_VOWEL_STARTS:
                    continue
            if vowel:
                head, *tail = arpa.split()
                phones.append(head if head[-1].isdigit() else head + stress)
                phones.extend(tail)
                stress = "0"
            else:
                phones.append(arpa)
            i = j
            break
        else:
            i += 1                       # length, aspiration, a glottal stop: no ARPAbet phone
    return phones


def _fold_arpabet(phonemes: str, language: str) -> str:
    words = [_arpabet_word(w) for w in phonemes.split()]
    return " | ".join(" ".join(w) for w in words if w)


# -- the `dizione` style (tiny-voca, loom.cpp ADR-133) -----------------------------------------------
#
# dizione (Apache-2.0, numpy only) is tiny-voca's own front end: 59 American English phones, stress as a
# mark before the phone it falls on, `<sp>` between words, punctuation as symbols, each SENTENCE wrapped
# `<bos> ... <eos>` -- the reference (`tiny_voca.infer.speak_text`) phonemizes and speaks a text one
# sentence at a time. When dizione is installed (and its data found: `DIZIONE_DATA`, which the model's
# own repo ships under `phonemizer/`), the door IS dizione, cut the reference's way. Without it, the
# default G2P's IPA is folded to espeak's conventions (length marks, stress before the vowel: dizione's
# spelling) and segmented into dizione's inventory, longest symbol first -- measurably worse (the card states both), and better than no text door.

_DIZIONE_PHONES = ("tʃ dʒ p b t d k ɡ f v θ ð s z ʃ ʒ h m n ŋ l ɹ j w ɾ ʔ "
                   "aɪɚ aɪə ɑːɹ ɔːɹ oːɹ ɪɹ ɛɹ ʊɹ əl n̩ eɪ aɪ ɔɪ oʊ aʊ iə oː "
                   "ə ɐ ɚ ᵻ ɪ iː i ɛ æ ʌ ɑː ɔː ɔ ʊ uː ɜː").split()
_DIZIONE_ORDER = sorted(_DIZIONE_PHONES, key=len, reverse=True)
_DIZIONE_PUNCTUATION = set(",.?!;:\"'()—…")
_SPACE = re.compile("[\\s" + chr(0xFEFF) + "]+")
_SENTENCE_END = re.compile(r"""(?:(?<=[.!?…])|(?<=[.!?…]["')\]]))\s+(?=["'(\[]?[A-Z0-9])""")
_NOT_AN_END = re.compile(r"(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$",
                         re.IGNORECASE | re.ASCII)
_TEXT_END = re.compile("[\\s" + chr(0xFEFF) + "]+$")
_CLOSING = "\"'”’)]»"


def dizione_sentences(text: str) -> list:
    """`tiny_voca.infer.sentences(finished(text))`: a full stop added to a text ending without a mark,
    then cut after sentence-final marks, never after a title, an initial or inside a number."""
    body = _TEXT_END.sub("", text)
    if body:
        last = _SPACE.split(body)[-1]
        if any(ch.isalnum() for ch in last):
            marked = last.rstrip(_CLOSING)[-1:] in tuple(".!?…,;:")
        else:
            marked = any(ch in ".!?…" for ch in last) or last in (",", ";", ":")
        text = text if marked else body + "."
    parts, current = [], ""
    for piece in _SENTENCE_END.split(_SPACE.sub(" ", text).strip()):
        current = f"{current} {piece}" if current else piece
        if not _NOT_AN_END.search(current):
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return [part for part in parts if any(ch.isalnum() for ch in part)]


def _dizione_native():
    try:
        import dizione
    except ImportError:
        return None
    return dizione


def _fold_dizione(phonemes: str, language: str) -> str:
    """One sentence's IPA -> `<bos> ... <eos>` in dizione's inventory: espeak's conventions first, then
    each word segmented longest symbol first, a stress mark carried onto the next phone. The syllabic
    `əl` and `n̩` are taken only where no vowel follows ("bottle", not the `əl` of "hello")."""
    folded = _fold_espeak(phonemes, language)
    words = []
    for word in folded.split():
        out, stress, i = [], "", 0
        while i < len(word):
            ch = word[i]
            if ch in _STRESS:
                stress = ch
                i += 1
                continue
            if ch in _DIZIONE_PUNCTUATION:
                if out:
                    words.append(out)
                    out = []
                words.append([ch])
                i += 1
                continue
            for symbol in _DIZIONE_ORDER:
                if word.startswith(symbol, i):
                    after = word[i + len(symbol): i + len(symbol) + 1]
                    if symbol in ("əl", "n̩") and after and (after in _STRESS or after in _VOWELS):
                        continue
                    out.append(stress + symbol)
                    stress = ""
                    i += len(symbol)
                    break
            else:
                i += 1                   # outside the inventory: dropped, as the table would drop it
        if out:
            words.append(out)
    tokens = ["<bos>"]
    for w in words:
        is_mark = len(w) == 1 and w[0] in _DIZIONE_PUNCTUATION
        if not is_mark and tokens[-1] not in ("<bos>",) and tokens[-1] not in _DIZIONE_PUNCTUATION:
            tokens.append("<sp>")
        tokens.extend(w)
    tokens.append("<eos>")
    return " ".join(tokens)


# -- the `kokoro` style (Paradee: loom-exporter paradee_export.py) ------------------------------------
#
# What Kokoro's own pipeline (`KPipeline`) writes, which is what a model distilled from it read in training:
# misaki for American and British English (one symbol per diphthong and affricate -- `A I W O Y`, British
# `Q` -- stress before the vowel, punctuation kept as symbols), and for every other language espeak's IPA as
# misaki's `EspeakG2P` hands it over: length and aspiration kept, tied pairs made one symbol. Upstream's
# runtime phonemizes English with misaki and Hindi, French and Spanish with Paradee's own Apache-2.0 G2P
# (`paradee.g2p`, trained to give espeak's output without espeak), so when those are installed the door IS
# them. Without them the default G2P runs between punctuation marks -- it drops marks, and the model reads a
# comma as a pause and a full stop as an end -- and each run is folded to the language's conventions, the
# marks put back after the word before them, as misaki writes them.

_KOKORO_MARKS = re.compile(r"\s*([,.!?;:—…]+)\s*")
_KOKORO_NATIVE: Dict[str, Optional[Callable[[str], str]]] = {}
# The tied pairs misaki's EspeakG2P compresses, for a language whose default G2P writes them as two symbols.
_KOKORO_TIES = (("dʒ", "ʤ"), ("tʃ", "ʧ"), ("ts", "ʦ"), ("dz", "ʣ"))
# Hindi: orthography2ipa's spellings to espeak's (`ɟ` and `c` for the palatal stops, `h` for the breathy
# glottal, no dental mark).
_KOKORO_HINDI = (("dʒ", "ɟ"), ("tʃ", "c"), ("ɦ", "h"), ("\u032a", ""))


def _kokoro_language(language: str) -> str:
    """`en-US`/`en-GB`/`fr-FR`/... -> `a`/`b` (misaki's American/British) or the bare language code."""
    code = language.replace("_", "-").lower()
    if code.startswith("en"):
        return "b" if code in ("en-gb", "en-uk") else "a"
    return code.split("-")[0]


def _kokoro_native(lang: str) -> Optional[Callable[[str], str]]:
    """The installed front end for `lang`, or None: misaki for English, Paradee's G2P for the rest."""
    if lang in _KOKORO_NATIVE:
        return _KOKORO_NATIVE[lang]
    fn = None
    try:
        if lang in ("a", "b"):
            from misaki import en

            british = lang == "b"

            def unknown(token):   # misaki's fallback for a word its lexicon lacks: the default G2P, folded
                return _fold_misaki(phonemize(token.text, language="en-GB" if british else "en-US"), "en"), 1

            g2p = en.G2P(trf=False, british=british, fallback=unknown, unk="")
            fn = lambda text: str(g2p(text)[0])
        else:
            from paradee.g2p import G2P

            own = G2P()
            fn = lambda text: str(own(text, lang))
    except Exception:            # not installed, or installed without its data: the folded default below
        fn = None
    _KOKORO_NATIVE[lang] = fn
    return fn


def _fold_kokoro(phonemes: str, language: str) -> str:
    lang = _kokoro_language(language)
    if lang in ("a", "b"):
        return _fold_misaki(phonemes, language)
    words = []
    for word in phonemes.split():
        word = _stress_before_vowel(word)
        if lang == "hi":
            word = _replace_all(word, _KOKORO_HINDI)
        else:
            word = _replace_all(word, _KOKORO_TIES)
        words.append(word)
    return " ".join(words)


def phonemize_kokoro(text: str, *, alphabet: str = "ipa", language: str = "en-US") -> str:
    """The `kokoro` style's phonemes for `text` (see the note above): the language's own front end when it
    is installed, else the default G2P between punctuation marks, folded, with the marks kept."""
    native = _kokoro_native(_kokoro_language(language))
    if native is not None:
        return native(text)
    out = []
    for i, part in enumerate(_KOKORO_MARKS.split(text)):
        if i % 2:                                     # a run of marks: after the word before it
            if out:
                out[-1] += part.replace(" ", "")
            continue
        if part.strip():
            out.append(_fold_kokoro(phonemize(part, alphabet=alphabet, language=language), language))
    return " ".join(out)


def phonemize_styled(text: str, style: str, *, alphabet: str = "ipa", language: str = "en") -> str:
    """The text door's phonemes for a model declaring `style`: the style's own front end when it has one
    and it is installed (dizione), else the registered G2P folded to the style."""
    if style == "kokoro":
        return phonemize_kokoro(text, alphabet=alphabet, language=language)
    if style == "dizione":
        sentences = dizione_sentences(text)
        native = _dizione_native()
        if native is not None:
            return " ".join(str(native.phonemize(s)) for s in sentences)
        return " ".join(fold(phonemize(s, alphabet=alphabet, language=language), style, language=language)
                        for s in sentences)
    return fold(phonemize(text, alphabet=alphabet, language=language), style, language=language)


#: `{style: fold(phonemes, language) -> phonemes}`, the styles a model may declare.
STYLES: Dict[str, Callable[[str, str], str]] = {"espeak": _fold_espeak, "misaki": _fold_misaki,
                                                "kokoro": _fold_kokoro,
                                                "arpabet": _fold_arpabet, "dizione": _fold_dizione}


def fold(phonemes: str, style: str, *, language: str = "en") -> str:
    """Rewrite a G2P's IPA into the conventions `style` names (see the note above). An unknown style is
    refused rather than passed through: a model declaring one this package does not know would
    otherwise get the unfolded string and sound wrong with nothing saying why."""
    if not style:
        return phonemes
    if style not in STYLES:
        raise LookupError(f"unknown phoneme style {style!r}; this loom-py folds to {sorted(STYLES)}. "
                          f"Upgrade loom-py-rt, or pass phonemes= in the model's own conventions.")
    return STYLES[style](phonemes, language)

def _load_default(alphabet: str):
    """`orthography2ipa`, if it is installed and the alphabet is one it produces.

    Imported on demand rather than at module import: this package is imported by everything, and the
    default provider pulls numpy and a language-data package behind it. A caller who never synthesizes
    never pays for it, and a broken install of it cannot stop `import loom` from working.
    """
    if alphabet != "ipa":
        return None
    try:
        import orthography2ipa
    except ImportError:
        return None

    def provider(text: str, language: str) -> str:
        # `transcribe(text, lang, *, search="greedy", beam_width=8, dialect_profile=None) -> str`,
        # read off the installed library rather than guessed. The defaults are deliberate here:
        #
        # SEARCH stays greedy, because `search="beam"` is not an improvement to buy. Measured across
        # widths 4/8/16/64 on English it returns the greedy string UNCHANGED -- the transduction has no
        # competing candidates to reorder -- so a beam would cost time per call and change nothing. An
        # earlier version of this comment said the beam was internal and its tie-break had to be matched
        # by the C++ port; both halves were wrong, and the port has one less thing to reproduce.
        #
        # DIALECT_PROFILE stays None because all fifteen shipped profiles are Portuguese/Galician;
        # there is no English one to pass.
        #
        # What DOES move the output is a lexicon, which is `set_lexicon` above and not a parameter here:
        # a language with one transcribes through the engine built over it.
        engine = _ENGINES.get(orthography2ipa.resolve(language)) if _ENGINES else None
        if engine is not None:
            return str(engine.transcribe(text))
        return str(orthography2ipa.transcribe(text, language))

    register(alphabet, provider)
    return provider
