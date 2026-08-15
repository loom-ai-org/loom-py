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

from typing import Callable, Dict

#: `{alphabet: callable(text, language) -> str}`. Keyed by alphabet rather than by package, because what
#: a model declares in `loom.text.phoneme_alphabet` is the alphabet it was trained on -- which provider
#: produced it is not something the model has an opinion about.
_PROVIDERS: Dict[str, Callable[[str, str], str]] = {}


def register(alphabet: str, phonemize: Callable[[str, str], str]) -> None:
    """Register a G2P for an alphabet, replacing any previous one.

    `phonemize(text, language) -> str` returns the phoneme string the model's own table will be asked to
    encode. Symbols outside that table are dropped by the vocabulary rather than refused, because a
    rule-based engine emits a superset of what any one checkpoint was trained on.
    """
    _PROVIDERS[alphabet] = phonemize


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
        # `transcribe(text, lang) -> str`, verified against the library's own README rather than
        # guessed. The beam search the engine runs is internal to it: one string comes back, so the
        # tie-break that must be pinned when this is ported to C++ lives inside orthography2ipa and not
        # in this call. That is worth knowing rather than reassuring -- a port that reimplements the
        # search without matching its ordering will disagree with this on the same sentence.
        return str(orthography2ipa.transcribe(text, language))

    register(alphabet, provider)
    return provider
