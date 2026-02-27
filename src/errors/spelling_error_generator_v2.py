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
    ("ą→on",      r"ą(?=[tdcnszśźć])",  "on"),
    ("ą→om",      r"ą(?=[bpm])",         "om"),
    ("ą→o",       r"ą$",                 "o"),
    ("ę→en",      r"ę(?=[tdcnszśźć])",  "en"),
    ("ę→em",      r"ę(?=[bpm])",         "em"),
    ("ę→e",       r"ę$",                 "e"),

    # ── Digraph confusions ──────────────────────────────────────────────
    ("ch→h",      r"ch",                 "h"),
    ("h→ch",      r"(?<!c)h",            "ch"),
    ("rz→ż",     r"rz",                 "ż"),
    ("ż→rz",     r"ż",                  "rz"),

    # ── ó / u ───────────────────────────────────────────────────────────
    ("ó→u",       r"ó",                  "u"),
    ("u→ó",       r"u",                  "ó"),

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


class SpellingErrorGeneratorV2(ErrorGenerator):
    """Introduces Polish spelling errors via phonetic rules + dictionary.

    For each word the generator:

    1. Checks the dictionary for an exact match (case-insensitive).
    2. If no match, collects all applicable rules, picks one at random,
       and applies it to a random occurrence within the word.
    3. Each word is modified with probability *rate*.

    Multi-word dictionary entries (e.g. ``na pewno → napewno``) are matched
    first, longest-phrase-first.

    Args:
        rate:  Probability (0.0–1.0) of modifying each eligible word.
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
        self._rng = random.Random(seed)
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
        # Phase 1: multi-word dictionary entries.
        for pattern, replacement in self._multi_word:
            text = self._replace_matches(text, pattern, replacement)

        # Phase 2: single-word processing (dict lookup → random rule).
        return self._process_words(text)

    # ── internals ───────────────────────────────────────────────────────

    def _replace_matches(
        self, text: str, pattern: re.Pattern[str], replacement: str,
    ) -> str:
        """Replace *pattern* matches in *text* with probability *rate*."""
        parts: list[str] = []
        prev = 0
        for m in pattern.finditer(text):
            parts.append(text[prev : m.start()])
            if self._rng.random() < self.rate:
                parts.append(replacement)
            else:
                parts.append(m.group())
            prev = m.end()
        parts.append(text[prev:])
        return "".join(parts)

    def _process_words(self, text: str) -> str:
        """Iterate over words, applying dict lookup or a random rule."""
        tokens = re.findall(r"\w+|\W+", text, flags=re.UNICODE)
        out: list[str] = []

        for tok in tokens:
            # Pass non-alphabetic tokens through unchanged.
            if not tok[0].isalpha():
                out.append(tok)
                continue

            # Modify with probability *rate*.
            if self._rng.random() >= self.rate:
                out.append(tok)
                continue

            out.append(self._modify_word(tok))

        return "".join(out)

    def _modify_word(self, word: str) -> str:
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
        _, pattern, repl = self._rng.choice(applicable)
        matches = list(pattern.finditer(low))
        m = self._rng.choice(matches)
        return word[: m.start()] + repl + word[m.end() :]
