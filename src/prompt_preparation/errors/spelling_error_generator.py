"""Rule-based Polish spelling error generator (v2).

Combines phonetic substitution rules with a fallback dictionary.

**Rules** cover common Polish spelling confusions:

- ó ↔ u, rz ↔ ż, ch ↔ h
- Nasal vowels: ą → on/om/o, ę → en/em/e
- Soft consonants: ci → ć, si → ś, zi → ź, ni → ń, dzi → dź
- Final consonant voicing confusion: g ↔ k, d ↔ t, b → p, w ↔ f, z → s
- Suffix simplification: -cji → -ci, -ii → -i
- Past tense nasals: -nął → -noł, -nęła → -neła
- ść → źć

**Dictionary** covers cases that cannot be expressed as simple rules:

- *nie-* prefix split/join (niedaleko ↔ nie daleko)
- Preposition merging (na pewno → napewno)
- Word boundary changes (naprawdę → na prawdę)
"""

import random
import re

from .base import ErrorGenerator
from ._rng import deterministic_seed

# ── Dictionary of irregular spelling errors ─────────────────────────────
# These cannot be expressed as simple character-substitution rules
# (word boundary changes, prefix splits/joins, irregular forms).
_SPELLING_DICT: dict[str, str] = {
    # ── Preposition / particle merging ──────────────────────────────────
    "na pewno": "napewno",
    "na razie": "narazie",
    "od razu": "odrazu",
    "po prostu": "poprostu",
    "co najmniej": "conajmniej",
    "na co dzień": "na codzień",
    "wszech czasów": "wszechczasów",
    "w ogóle": "wogóle",
    "w głąb": "wgłąb",
    "w skład": "wskład",
    "z powrotem": "spowrotem",
    "na przykład": "naprzykład",
    "nie lada": "nielada",
    "nie byle": "niebyle",
    "można by": "możnaby",
    "trzeba by": "trzebaby",
    "dookoła": "do okoła",
    "z wyjątkiem": "za wyjątkiem",

    # ── Word splitting (joined → split) ─────────────────────────────────
    "naprawdę": "na prawdę",
    "nieprawda": "nie prawda",
    "naprzeciwko": "na przeciwko",
    "naprzeciw": "na przeciw",
    "niemniej": "nie mniej",
    "pośrodku": "po środku",

    # ── Prefix z-/s- confusion ──────────────────────────────────────────
    "stąd": "z tąd",
    "stamtąd": "z tamtąd",
    "znikąd": "z nikąd",
    "sprzed": "z przed",
    "znad": "z nad",
    "spod": "z pod",
    "spomiędzy": "z pomiędzy",
    "spośród": "z pośród",
    "spoza": "z poza",
    "wśród": "w śród",
    "wzdłuż": "w zdłuż",
    "wskutek": "w skutek",
    "zza": "z za",
    "skąd": "zkąd"
}

# ── Rule definitions ────────────────────────────────────────────────────
# (name, regex_pattern, replacement)
# Patterns are matched against *lowercased* individual word tokens.
# ``$`` = end of word, ``^`` = start of word.

_RULE_DEFS: list[tuple[str, str, str]] = [
    # ── Nasal vowels ────────────────────────────────────────────────────
    ("ą→on",      r"ą(?=[tdcnszśźćkg])",  "on"),
    ("ą→om",      r"ą(?=[bpm])",         "om"),
    ("ą→o",       r"ą$",                 "o"),
    ("ę→en",      r"ę(?=[tdcnszśźćkg])",  "en"),
    ("ę→em",      r"ę(?=[bpm])",         "em"),
    ("ę→e",       r"ę$",                 "e"),

    # ── Digraph confusions ──────────────────────────────────────────────
    ("ch→h",      r"ch",                 "h"),
    ("h→ch",      r"(?<!c)h",            "ch"),
    ("rz→ż",     r"rz",                 "ż"),
    ("ż→rz",     r"ż",                  "rz"),

    # ── ó / u ───────────────────────────────────────────────────────────
    ("ó→u",       r"ó",                  "u"),

    # ── Soft consonants (before vowels) ─────────────────────────────────
    ("ci→ć",      r"ci(?=[aeouyąęó])",   "ć"),
    ("si→ś",      r"si(?=[aeouyąęó])",   "ś"),
    ("zi→ź",      r"zi(?=[aeouyąęó])",   "ź"),
    ("ni→ń",      r"ni(?=[aeouyąęó])",   "ń"),
    ("dzi→dź",    r"dzi(?=[aeouyąęó])",  "dź"),

    # ── ść → źć ─────────────────────────────────────────────────────────
    ("ść→źć",     r"ść",                 "źć"),

    # ── Final consonant confusion (bidirectional) ───────────────────────
    ("g→k",       r"g$",                 "k"),
    ("k→g",       r"k$",                 "g"),
    ("d→t",       r"d$",                 "t"),
    ("t→d",       r"t$",                 "d"),
    ("b→p",       r"b$",                 "p"),
    ("w→f",       r"w$",                 "f"),
    ("f→w",       r"f$",                 "w"),
    ("z→s",       r"z$",                 "s"),

    # ── Common suffix patterns ──────────────────────────────────────────
    ("nął→noł",   r"nął",               "noł"),
    ("nęła→neła", r"nęła",              "neła"),
    ("cji→ci",    r"cji$",              "ci"),
    ("ii→i",      r"ii$",               "i"),
]

# Pre-compile all rule patterns.
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (name, re.compile(pat, re.UNICODE), repl)
    for name, pat, repl in _RULE_DEFS
]

#: Set of all available rule names.
RULE_NAMES: frozenset[str] = frozenset(name for name, _, _ in _RULE_DEFS)


class SpellingErrorGenerator(ErrorGenerator):
    """Introduces Polish spelling errors via phonetic rules + dictionary.

     For each input text the generator:

     1. Computes a target number of spelling errors from *rate* and word count.
     2. Tries to spend this budget first on multi-word dictionary entries.
     3. Spends remaining budget on eligible single words (dict/rules),
         allowing multiple non-overlapping changes per word.

    Multi-word dictionary entries (e.g. ``na pewno → napewno``) are matched
    first, longest-phrase-first.

    Args:
        rate:  Target fraction (0.0–1.0) of words to modify.
        seed:  RNG seed for reproducibility.
    """

    def __init__(
        self,
        rate: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be between 0.0 and 1.0, got {rate!r}")

        self.rate = rate
        self._seed = seed if seed is not None else 0
        self._salt = "spelling"
        self._rules = list(_RULES)

        # Build dictionary lookup structures.
        self._multi_word: list[tuple[re.Pattern[str], str]] = []
        self._single_word: dict[str, str] = {}

        for correct, incorrect in _SPELLING_DICT.items():
            if " " in correct:
                pat = re.compile(
                    rf"\b{re.escape(correct)}\b",
                    re.IGNORECASE | re.UNICODE,
                )
                self._multi_word.append((pat, incorrect))
            else:
                self._single_word[correct.lower()] = incorrect

        # Longer phrases matched first.
        self._multi_word.sort(key=lambda x: -len(x[0].pattern))

    # ── public API ──────────────────────────────────────────────────────

    def apply(self, text: str) -> str:
        rng_seed = deterministic_seed(self._seed, text, self._salt)
        rng = random.Random(rng_seed)

        # Budget is counted in modified words, not in random Bernoulli trials.
        target = int(self._round_it(self.rate * len([word for word in text.split(" ")])))
        if target <= 0:
            return text

        # Phase 1: spend budget on multi-word dictionary entries.
        remaining = target
        for pattern, replacement in self._multi_word:
            if remaining <= 0:
                break
            text, used = self._replace_matches_budget(text, pattern, replacement, remaining, rng)
            remaining -= used

        if remaining <= 0:
            return text

        # Phase 2: spend remaining budget on eligible single words.
        return self._process_words_budget(text, remaining, rng)


    # ── internals ───────────────────────────────────────────────────────
    @staticmethod
    def _round_it(number: float, position: int = 0) -> float:
        factor = 10 ** position
        return int(number * factor + 0.5) / factor
    
    @staticmethod
    def _count_match_words(text: str) -> int:
        return sum(1 for part in text.split() if part and part[0].isalpha())

    def _replace_matches_budget(
        self,
        text: str,
        pattern: re.Pattern[str],
        replacement: str,
        budget: int,
        rng: random.Random,
    ) -> tuple[str, int]:
        """Replace non-overlapping matches while consuming at most *budget* words."""
        if budget <= 0:
            return text, 0

        matches = list(pattern.finditer(text))
        if not matches:
            return text, 0

        order = list(range(len(matches)))
        rng.shuffle(order)

        selected: set[int] = set()
        consumed = 0
        for idx in order:
            m = matches[idx]
            cost = max(1, self._count_match_words(m.group()))
            if consumed + cost > budget:
                continue
            selected.add(idx)
            consumed += cost
            if consumed >= budget:
                break

        if not selected:
            return text, 0

        parts: list[str] = []
        prev = 0
        for idx, m in enumerate(matches):
            parts.append(text[prev : m.start()])
            if idx in selected:
                parts.append(replacement)
            else:
                parts.append(m.group())
            prev = m.end()
        parts.append(text[prev:])
        return "".join(parts), consumed

    def _get_all_changes(self, word: str) -> list[tuple[int, int, str]]:
        """Collect all currently applicable changes for one word token."""
        if len(word) <= 2:
            return []

        low = word.lower()
        changes: list[tuple[int, int, str]] = []

        # Whole-word dictionary replacement.
        if low in self._single_word:
            changes.append((0, len(word), self._single_word[low]))

        # Rule-based replacements at all match positions.
        for _, pattern, repl in self._rules:
            for m in pattern.finditer(low):
                changes.append((m.start(), m.end(), repl))

        return changes

    def _process_words_budget(self, text: str, budget: int, rng: random.Random) -> str:
        """Apply up to *budget* individual non-overlapping changes."""
        tokens = re.findall(r"\w+|\W+", text, flags=re.UNICODE)
        if budget <= 0:
            return "".join(tokens)

        # token index -> list of candidate changes (start, end, replacement)
        pending: dict[int, list[tuple[int, int, str]]] = {}
        for i, tok in enumerate(tokens):
            if tok and tok[0].isalpha():
                changes = self._get_all_changes(tok)
                if changes:
                    pending[i] = changes

        if not pending:
            return "".join(tokens)

        while budget > 0 and pending:
            token_idx = rng.choice(list(pending.keys()))
            changes = pending[token_idx]

            chosen_idx = rng.randrange(len(changes))
            start, end, repl = changes[chosen_idx]

            tok = tokens[token_idx]
            tokens[token_idx] = tok[:start] + repl + tok[end:]
            budget -= 1

            # Keep only non-overlapping candidates and shift those after edit.
            delta = len(repl) - (end - start)
            next_changes: list[tuple[int, int, str]] = []
            for i, (s, e, r) in enumerate(changes):
                if i == chosen_idx:
                    continue
                if e <= start:
                    next_changes.append((s, e, r))
                elif s >= end:
                    next_changes.append((s + delta, e + delta, r))
                # overlapping change is dropped

            if next_changes:
                pending[token_idx] = next_changes
            else:
                del pending[token_idx]

        return "".join(tokens)

    def _can_modify_word(self, word: str) -> bool:
        if len(word) <= 2:
            return False
        low = word.lower()
        if low in self._single_word:
            return True
        for _, pattern, _ in self._rules:
            if pattern.search(low):
                return True
        return False

    def _modify_word(self, word: str, rng: random.Random) -> str:
        """Try dict, then rules.  Returns modified or original word."""
        low = word.lower()

        # 1) Dictionary lookup.
        if low in self._single_word:
            return self._single_word[low]

        # 2) Collect applicable rules.
        applicable: list[tuple[str, re.Pattern[str], str]] = []
        for name, pattern, repl in self._rules:
            if pattern.search(low):
                applicable.append((name, pattern, repl))

        if not applicable:
            return word  # nothing to change

        # 3) Pick a random rule, apply to a random match position.
        _, pattern, repl = rng.choice(applicable)
        matches = list(pattern.finditer(low))
        m = rng.choice(matches)
        return word[: m.start()] + repl + word[m.end() :]
