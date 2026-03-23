import re

from .base import ErrorGenerator

_INNER_PUNCTUATION_CHARS = (
    ",;:"
    "\u2013\u2014" # pauza i półpauza
    "\"'" # proste cudzysłowy
    "\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u00ab\u00bb" # polskie cudzysłowy
    "()[]{}<>" # nawiasy
    "\u2026" # wielokropek
)
_ALL_PUNCTUATION_RE = re.compile(
    f"[{re.escape('.!?' + _INNER_PUNCTUATION_CHARS)}]"
)
_INNER_PUNCTUATION_RE = re.compile(f"[{re.escape(_INNER_PUNCTUATION_CHARS)}]")

class PunctuationAllErrorGenerator(ErrorGenerator):
    """Removes ALL punctuation and converts text to lowercase."""

    def apply(self, text: str) -> str:
        return _ALL_PUNCTUATION_RE.sub("", text.lower())


class PunctuationInnerErrorGenerator(ErrorGenerator):
    """Removes only inner-sentence punctuation."""

    def apply(self, text: str) -> str:
        return _INNER_PUNCTUATION_RE.sub("", text)