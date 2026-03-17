import re

from .base import ErrorGenerator

_ALL_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_INNER_PUNCTUATION_RE = re.compile(r'[,;:\-–—"\'„\u201c«»()\[\]{}…/\\]', re.UNICODE)


class PunctuationAllErrorGenerator(ErrorGenerator):
    """Removes ALL punctuation and lowercases."""

    def apply(self, text: str) -> str:
        return _ALL_PUNCTUATION_RE.sub("", text.lower())




class PunctuationInnerErrorGenerator(ErrorGenerator):
    """Removes only inner-sentence punctuation.

    Sentence-ending marks (. ! ?) are preserved and capital letters
    at the beginning of each sentence are kept intact.
    """

    def apply(self, text: str) -> str:
        sentences = _SENTENCE_SPLIT_RE.split(text)
        cleaned = []
        for sentence in sentences:
            cleaned.append(_INNER_PUNCTUATION_RE.sub("", sentence))
        return " ".join(cleaned)
