from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .base import ErrorGenerator

# ---------------------------------------------------------------------------
# Polish keyboard data (QWERTY Programmers layout)
# ---------------------------------------------------------------------------

_KEYBOARD: list[list[str]] = [
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm'],
]

_ALTGR_KEYBOARD: list[list[str]] = [
    ['q', 'w', 'ę', 'r', 't', 'y', 'u', 'i', 'ó', 'p'],
    ['ą', 'ś', 'd', 'f', 'g', 'h', 'j', 'k', 'ł'],
    ['ż', 'ź', 'ć', 'v', 'b', 'ń', 'm'],
]

_LEFT_RIGHT: dict[str, list[str]] = {
    'left': ['q', 'w', 'e', 'r', 't', 'a', 's', 'd', 'f', 'g', 'z', 'x', 'c', 'v', 'b',
             'ę', 'ą', 'ś', 'ć', 'ż', 'ź'],
    'right': ['y', 'u', 'i', 'o', 'p', 'h', 'j', 'k', 'l', 'n', 'm',
              'ó', 'ł', 'ń'],
}

_DIACRITICAL_MAP: dict[str, str] = {
    "ą": "a", "ę": "e", "ś": "s", "ć": "c",
    "ż": "z", "ź": "x", "ł": "l", "ó": "o", "ń": "n",
}

_IGNORE_SET: set[str] = {
    'zero', 'jeden', 'jedna', 'jedno', 'dwa', 'dwie', 'trzy', 'cztery', 'pięć', 'sześć', 'siedem', 'osiem', 'dziewięć',
    'dziesięć', 'jedenaście', 'dwanaście', 'trzynaście', 'czternaście', 'piętnaście', 'szesnaście', 'siedemnaście', 'osiemnaście', 'dziewiętnaście',
    'dwadzieścia', 'trzydzieści', 'czterdzieści', 'pięćdziesiąt', 'sześćdziesiąt', 'siedemdziesiąt', 'osiemdziesiąt', 'dziewięćdziesiąt',
    'sto', 'dwieście', 'tysiąc', 'milion', 'miliard',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
}

# ---------------------------------------------------------------------------
# Typo generation engine
# ---------------------------------------------------------------------------

TypoFunc = Callable[[str, "_TypoEngine"], Tuple[str, Optional[Tuple[int, ...]]]]

_DEFAULT_TYPO_DISTRIBUTION: Dict[str, float] = {
    "delete": 0.28,
    "insert": 0.15,
    "replace": 0.28,
    "transpose": 0.28,
}


@dataclass
class _TypoEngine:
    """Keyboard-proximity typo engine for the Polish QWERTY layout."""

    use_excluding_set: bool = True
    typo_distribution: Dict[str, float] = field(default_factory=lambda: _DEFAULT_TYPO_DISTRIBUTION.copy())
    horizontal_vs_vertical: Tuple[float, float] = (9.0, 1.0)

    def __post_init__(self) -> None:
        self.keyboard = _KEYBOARD
        self.left_right = _LEFT_RIGHT
        self.ignore_set = _IGNORE_SET
        self.diacritical_map = _DIACRITICAL_MAP
        self.altgr_keyboard = _ALTGR_KEYBOARD
        self._normalize_distribution()

    def _normalize_distribution(self) -> None:
        total = sum(self.typo_distribution.values())
        if total <= 0:
            raise ValueError("typo_distribution must have positive total mass.")
        for k in self.typo_distribution:
            self.typo_distribution[k] /= total

    def _get_neighbours_with_orientation(self, char: str) -> List[str]:
        h_weight, v_weight = self.horizontal_vs_vertical

        base_char = self.diacritical_map.get(char)
        if base_char is not None and self.altgr_keyboard is not None:
            grid = self.altgr_keyboard
        else:
            grid = self.keyboard

        positions = []
        for r_idx, row in enumerate(grid):
            for c_idx, k in enumerate(row):
                if k == char:
                    positions.append((r_idx, c_idx))

        if not positions:
            return []

        candidates: List[str] = []
        for r_idx, c_idx in positions:
            if c_idx > 0:
                left = grid[r_idx][c_idx - 1]
                if left:
                    candidates.extend([left] * max(1, int(round(h_weight))))
            if c_idx < len(grid[r_idx]) - 1:
                right = grid[r_idx][c_idx + 1]
                if right:
                    candidates.extend([right] * max(1, int(round(h_weight))))

            if r_idx > 0 and c_idx < len(grid[r_idx - 1]):
                up = grid[r_idx - 1][c_idx]
                if up:
                    candidates.extend([up] * max(1, int(round(v_weight))))
            if r_idx < len(grid) - 1 and c_idx < len(grid[r_idx + 1]):
                down = grid[r_idx + 1][c_idx]
                if down:
                    candidates.extend([down] * max(1, int(round(v_weight))))

        return candidates

    @staticmethod
    def _round_it(number: float, position: int = 0) -> float:
        factor = 10 ** position
        return int(number * factor + 0.5) / factor

    def _get_weights_of_idx(self, word: str) -> List[float]:
        n = len(word)
        if n == 1:
            return [0.0]
        elif n == 2:
            return [0.0, 1.0]

        weights = [0.0]
        for i in range(0, n - 1):
            weight = 0.1 + (0.2 - 0.1) * (i / (n - 2))
            weights.append(weight)

        weights = [self._round_it(p / sum(weights), 2) for p in weights]
        return weights

    def _belongs_to_hand(self, char: str) -> Optional[str]:
        if char in self.left_right["left"]:
            return "left"
        if char in self.left_right["right"]:
            return "right"
        return None

    def _replace(self, word: str) -> Tuple[str, Optional[Tuple[int, ...]]]:
        word_list = list(word)
        weights = self._get_weights_of_idx(word)
        idx, char = random.choices(list(enumerate(word_list)), weights=weights, k=1)[0]

        neighbours = self._get_neighbours_with_orientation(char.lower())
        if neighbours:
            replacement = random.choice(neighbours)
            if char.isupper():
                replacement = replacement.upper()

            for i in range(idx - 1, -1, -1):
                if word_list[idx] != word_list[i]:
                    if (idx - i) == 2:
                        word_list[i + 1] = replacement
                        word_list[idx] = replacement
                        return "".join(word_list), (idx,)
                    break

            if idx != len(word_list) - 1 and word_list[idx] == word_list[idx + 1]:
                word_list[idx + 1] = replacement

            word_list[idx] = replacement

        return "".join(word_list), (idx,)

    def _delete(self, word: str) -> Tuple[str, Optional[Tuple[int, ...]]]:
        word_list = list(word)
        weights = self._get_weights_of_idx(word)
        idx, _ = random.choices(list(enumerate(word_list)), weights=weights, k=1)[0]
        del word_list[idx]
        return "".join(word_list), (idx,)

    def _insert(self, word: str) -> Tuple[str, Optional[Tuple[int, ...]]]:
        word_list = list(word)
        weights = self._get_weights_of_idx(word)
        idx, char = random.choices(list(enumerate(word_list)), weights=weights, k=1)[0]

        neighbours = self._get_neighbours_with_orientation(char.lower())
        if neighbours:
            inserted = random.choice(neighbours)
            if char.isupper():
                inserted = inserted.upper()
            word_list.insert(idx + 1, inserted)

        return "".join(word_list), (idx + 1,)

    def _transpose(self, word: str) -> Tuple[str, Optional[Tuple[int, ...]]]:
        word_list = list(word[1:])
        if len(word) <= 1:
            return word, None

        transposable: List[Tuple[int, int]] = []

        for i in range(len(word_list) - 1):
            first_hand = self._belongs_to_hand(word_list[i].lower())
            second_hand = self._belongs_to_hand(word_list[i + 1].lower())
            if (first_hand and second_hand and first_hand != second_hand) or word_list[i + 1] == " ":
                transposable.append((i, i + 1))

        if not transposable:
            return word, None

        weights = self._get_weights_of_idx(transposable)
        if sum(weights) == 0:
            return word, None

        first, second = random.choices(transposable, weights=weights, k=1)[0]
        word_list[first], word_list[second] = word_list[second], word_list[first]

        return word[0] + "".join(word_list), (first, second)

    def _is_valid_operation(self, word_history, idxs, typo_function: TypoFunc):
        if len(word_history) == 1:
            return True

        for prev_operation, prev_idx in word_history[1:]:
            if prev_operation is self._replace:
                if typo_function is self._insert and (prev_idx[0] + 1 == idxs[0]):
                    return False
                if (prev_idx == idxs) or any(i in idxs for i in prev_idx):
                    return False

            if prev_operation is self._delete:
                if typo_function is self._transpose and (prev_idx[0] == idxs[1]):
                    return False

            if prev_operation is self._insert:
                if (typo_function in {self._replace, self._delete}) and (
                    (prev_idx == idxs) or (prev_idx[0] - 1 == idxs[0])
                ):
                    return False
                elif (typo_function is self._insert) and (
                    (prev_idx == idxs) or (prev_idx[0] + 1 == idxs[0])
                ):
                    return False
                elif (typo_function is self._transpose) and (
                    (prev_idx == idxs) or any(i in idxs for i in prev_idx)
                ):
                    return False

            if prev_operation is self._transpose:
                if (typo_function in {self._transpose, self._insert, self._delete}) and (
                    (prev_idx == idxs) or any(i in idxs for i in prev_idx)
                ):
                    return False

        return True

    def insert_typos(
        self,
        text: str,
        typo_rate: float,
        max_tries: int = 1000,
    ) -> str:
        """Insert typos into *text*. *typo_rate* is the fraction of words to corrupt."""
        words = [[word] for word in text.split(" ")]
        typos_target = self._round_it(typo_rate * len(words))

        if typos_target < 0.5:
            return text

        weights = [math.sqrt(len(word[0])) for word in words]
        if sum(weights) == 0:
            return text
        weights = [self._round_it(w / sum(weights), 2) for w in weights]

        typoed = 0
        tries = 0
        new_word = ""
        ignore = True
        is_valid = False

        typo_funcs = {
            "delete": self._delete,
            "insert": self._insert,
            "replace": self._replace,
            "transpose": self._transpose,
        }

        typo_names = list(self.typo_distribution.keys())
        typo_weights = [self.typo_distribution[name] for name in typo_names]

        while typoed < typos_target and tries <= max_tries:
            while ignore and tries <= max_tries:
                tries += 1
                word_idx, word_info = random.choices(
                    list(enumerate(words)), weights=weights, k=1
                )[0]
                word = word_info[0]

                if len(word) <= 1:
                    ignore = True
                elif self.use_excluding_set and any(
                    str(i) in word.lower() for i in self.ignore_set
                ):
                    ignore = True
                else:
                    ignore = False

            if tries > max_tries:
                break

            while ((new_word == word) or not is_valid) and tries <= max_tries:
                tries += 1
                typo_name = random.choices(typo_names, weights=typo_weights, k=1)[0]
                typo_function = typo_funcs[typo_name]

                if (typo_function is self._transpose
                    and word_idx != len(words) - 1
                    and " " not in word):
                    word = word + " "

                new_word, idxs = typo_function(word)
                if new_word != word:
                    is_valid = self._is_valid_operation(words[word_idx], idxs, typo_function)

            if tries > max_tries:
                break

            weights[word_idx] *= 0.5
            words[word_idx][0] = new_word
            words[word_idx].append((typo_function, idxs))

            typoed += 1
            ignore = True
            is_valid = False
            new_word = ""

        final_words = [w[0] for w in words]
        result = ""
        for w in final_words:
            if " " in w:
                result += w
            else:
                result += (w + " ")
        return result.strip()


# ---------------------------------------------------------------------------
# Public ErrorGenerator subclass
# ---------------------------------------------------------------------------

class TypoErrorGenerator(ErrorGenerator):
    """Introduces realistic keyboard-based typos (delete, insert, replace, transpose)
    using Polish QWERTY keyboard-proximity data.

    Args:
        typo_rate: Fraction of words to corrupt (0.0–1.0). Defaults to 0.15.
        seed: Optional integer seed for reproducible results.
    """

    def __init__(
        self,
        typo_rate: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= typo_rate <= 1.0:
            raise ValueError(f"typo_rate must be between 0.0 and 1.0, got {typo_rate!r}")
        self._engine = _TypoEngine()
        self._typo_rate = typo_rate
        if seed is not None:
            random.seed(seed)

    def apply(self, text: str) -> str:
        return self._engine.insert_typos(text, typo_rate=self._typo_rate)
