import re

from .base import ErrorGenerator

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)


class PunctuationErrorGenerator(ErrorGenerator):
    """Removes all punctuation characters from the text."""

    def apply(self, text: str) -> str:
        return _PUNCTUATION_RE.sub("", text)
