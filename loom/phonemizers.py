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
import warnings
from typing import Callable, Dict, Optional

#: `{alphabet: callable(text, language) -> str}`. Keyed by alphabet rather than by package, because what
#: a model declares in `loom.text.phoneme_alphabet` is the alphabet it was trained on -- which provider
#: produced it is not something the model has an opinion about.
_PROVIDERS: Dict[str, Callable[[str, str], str]] = {}

#: `{language: source}` handed to `orthography2ipa.register_lexicon`, newest wins. Kept here rather than
#: passed per call because registration is process-global in that library and the parsed lexicon is
#: cached there for the process, so naming a source once is the whole interaction.
_LEXICONS: Dict[str, str] = {}


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
    `hf://<repo>/<path>` id, all three resolved by orthography2ipa itself. It is read here rather than
    on the first transcription for that language, so that a file which yields nothing is reported at
    the line that named it -- see WHY IT IS VERIFIED HERE below.

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
    lexicon registered for either is found. Registering under an unresolved tag is silent when it is
    wrong -- the lexicon simply never loads -- which is why this does the resolution rather than passing
    the caller's string through.

    **WHY IT IS VERIFIED HERE.** The parse skips, without complaint, every line that is not exactly
    `word<TAB>ipa` -- so a file of the wrong SHAPE loads as an empty overlay, and an empty overlay is
    byte-identical to no overlay at all: registration succeeds, transcription falls back to the rules,
    and the audio comes back unstressed with nothing anywhere saying why. That is not hypothetical.
    Running the documented `sed` over the GitHub *blob* page for ipa-dict instead of the raw file
    yields 700 lines of HTML, zero of them entries, and the only symptom is that `set_lexicon` appears
    not to be taken into account. So the lexicon is loaded now and a zero-entry one warns. The parse is
    not extra work, only earlier -- orthography2ipa caches it, and the first transcription would have
    paid for it anyway; what does move is the network fetch behind a URL or an `hf://` id.

    Raises `LookupError` when orthography2ipa is not installed, because a lexicon set on a provider that
    does not exist would otherwise be accepted and never applied. A source that cannot be READ raises
    whatever reading it raises -- `FileNotFoundError` for a path that is not there, a fetch error for a
    URL or an `hf://` id. Those were always raised; verifying here moves them off the first synthesis
    call and onto the line holding the wrong path.
    """
    try:
        import orthography2ipa
    except ImportError:
        raise LookupError(
            "a lexicon configures the default phonemizer, and orthography2ipa is not installed. "
            "Install it with `pip install \"loom-py-rt[phonemes]\"`. A provider registered with "
            "`loom.phonemizers.register(...)` brings its own pronunciations and needs none of this."
        ) from None

    code = orthography2ipa.resolve(language)
    if source is None:
        _LEXICONS.pop(code, None)
        return
    _LEXICONS[code] = str(source)
    orthography2ipa.register_lexicon(code, str(source))
    if not orthography2ipa.get_lexicon(code):
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
        # What DOES move the output is a lexicon, which is `set_lexicon` above and not a parameter here.
        return str(orthography2ipa.transcribe(text, language))

    register(alphabet, provider)
    return provider
