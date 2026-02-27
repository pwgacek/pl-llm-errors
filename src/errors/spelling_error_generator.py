import json
import random
import re
from pathlib import Path

from .base import ErrorGenerator

_DEFAULT_DICT = Path(__file__).parent / "assets" / "spelling_dict.json"


def _build_pattern(correct: str) -> re.Pattern[str]:
    """Build a whole-word-boundary regex for *correct* (works for phrases too)."""
    escaped = re.escape(correct)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE | re.UNICODE)


class SpellingErrorGenerator(ErrorGenerator):
    """Introduces spelling errors by replacing correct Polish words/phrases
    with their common misspellings, as defined in a JSON dictionary.

    The dictionary is a JSON object mapping correct forms to incorrect ones::

        {
            "na pewno": "napewno",
            ...
        }

    Matching is case-insensitive; the replacement is always written exactly as
    specified in the dictionary (no case-preservation).

    Args:
        dict_path: Path to the JSON spelling dictionary.
                   Defaults to ``spelling_dict.json`` in the same directory.
        rate: Probability (0.0–1.0) that any matching occurrence is replaced.
              Defaults to 1.0 (always replace).
        seed: Optional integer seed for reproducible results.
    """

    def __init__(
        self,
        dict_path: Path | str = _DEFAULT_DICT,
        rate: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be between 0.0 and 1.0, got {rate!r}")
        self.rate = rate
        self._rng = random.Random(seed)

        with open(dict_path, encoding="utf-8") as f:
            raw: dict[str, str] = json.load(f)

        # Sort by length descending so longer phrases are matched before
        # shorter substrings (e.g. "na pewno" before "pewno").
        self._rules: list[tuple[re.Pattern[str], str]] = [
            (_build_pattern(correct), incorrect)
            for correct, incorrect in sorted(raw.items(), key=lambda kv: -len(kv[0]))
        ]

    def apply(self, text: str) -> str:
        for pattern, incorrect in self._rules:
            text = self._replace_with_rate(text, pattern, incorrect)
        return text

    def _replace_with_rate(
        self,
        text: str,
        pattern: re.Pattern[str],
        replacement: str,
    ) -> str:
        result: list[str] = []
        prev_end = 0
        for match in pattern.finditer(text):
            result.append(text[prev_end : match.start()])
            if self._rng.random() < self.rate:
                result.append(replacement)
            else:
                result.append(match.group())
            prev_end = match.end()
        result.append(text[prev_end:])
        return "".join(result)
