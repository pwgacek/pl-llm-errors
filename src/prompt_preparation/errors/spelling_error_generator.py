"""Rule-based Polish spelling error generator.

Applies dictionary replacements first, then rule-based substitutions.
"""

import random
import re

from .base import ErrorGenerator
from ._rng import deterministic_seed

# Irregular spelling mappings.

_SPELLING_DICT: dict[str, str] = {
    # ── 1. Zrosty przyimków i partykuł (rozdzielone -> złączone) ────────
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

    # ── 2. Rozdzielanie słów (złączone -> rozdzielone) ──────────────────
    "naprawdę": "na prawdę",
    "naprzeciwko": "na przeciwko",
    "naprzeciw": "na przeciw",
    "niemniej": "nie mniej",
    "pośrodku": "po środku",
    "dookoła": "do okoła",
    "wśród": "w śród",
    "wzdłuż": "w zdłuż",
    "wskutek": "w skutek",
    "zza": "z za",

    # ── 3. Przedrostki z-/s- (mylenie zapisu fonetycznego) ──────────────
    "stąd": "z tąd",
    "stamtąd": "z tamtąd",
    "znikąd": "z nikąd",
    "znad": "z nad",
    "sprzed": "z przed",
    "spod": "z pod",
    "spomiędzy": "z pomiędzy",
    "spośród": "z pośród",
    "spoza": "z poza",
    "skąd": "zkąd",

    # ── 4. Pisownia z "NIE": Przymiotniki/Przysłówki (złącz. -> rozdz.) ─
    # Niezależny, Niebezpieczny, Niewielki
    "niezależny": "nie zależny", "niezależna": "nie zależna", "niezależne": "nie zależne", "niezależnie": "nie zależnie",
    "niebezpieczny": "nie bezpieczny", "niebezpieczna": "nie bezpieczna", "niebezpieczne": "nie bezpieczne", "niebezpiecznie": "nie bezpiecznie",
    "niewielki": "nie wielki", "niewielka": "nie wielka", "niewielkie": "nie wielkie", "niewiele": "nie wiele",
    
    # Niedobry, Niezły, Niedługi
    "niedobry": "nie dobry", "niedobra": "nie dobra", "niedobre": "nie dobre", "niedobrze": "nie dobrze",
    "niezły": "nie zły", "niezła": "nie zła", "niezłe": "nie złe", "nieźle": "nie źle",
    "niedługi": "nie długi", "niedługa": "nie długa", "niedługie": "nie długie", "niedługo": "nie długo",
    
    # Inne popularne przymiotniki i przysłówki
    "niełatwy": "nie łatwy", "niełatwa": "nie łatwa", "niełatwe": "nie łatwe", "niełatwo": "nie łatwo",
    "nietrudny": "nie trudny", "nietrudna": "nie trudna", "nietrudne": "nie trudne", "nietrudno": "nie trudno",
    "niezwykły": "nie zwykły", "niezwykła": "nie zwykła", "niezwykłe": "nie zwykłe", "niezwykle": "nie zwykle",
    "niesamowity": "nie samowity", "niesamowita": "nie samowita", "niesamowite": "nie samowite",
    "nieważny": "nie ważny", "nieważna": "nie ważna", "nieważne": "nie ważne",
    "niepotrzebny": "nie potrzebny", "niepotrzebna": "nie potrzebna", "niepotrzebne": "nie potrzebne", "niepotrzebnie": "nie potrzebnie",
    "niepewny": "nie pewny", "niepewna": "nie pewna", "niepewne": "nie pewne", "niepewnie": "nie pewnie",
    "nieznany": "nie znany", "nieznana": "nie znana", "nieznane": "nie znane",
    "niejasny": "nie jasny", "niejasna": "nie jasna", "niejasne": "nie jasne",
    "nieobecny": "nie obecny", "nieobecna": "nie obecna", "nieobecne": "nie obecne",
    "nielegalny": "nie legalny", "nielegalna": "nie legalna", "nielegalne": "nie legalne",
    
    # Rzeczowniki
    "nieprawda": "nie prawda", "nieporozumienie": "nie porozumienie", "niebezpieczeństwo": "nie bezpieczeństwo",

    # ── 5. Pisownia z "NIE": Czasowniki (rozdzielone -> złączone) ───────
    # Wiedzieć / Mieć
    "nie wiem": "niewiem", "nie wiemy": "niewiemy", "nie wiecie": "niewiecie", "nie wiedzą": "niewiedzą", "nie wiedział": "niewiedział", "nie wiedziała": "niewiedziała",
    "nie ma": "niema", "nie mam": "niemam", "nie masz": "niemasz", "nie mamy": "niemamy", "nie macie": "niemacie", "nie mają": "niemają",
    "nie miał": "niemiał", "nie miała": "niemiała", "nie mieli": "niemieli", "nie miałem": "niemiałem", "nie miałam": "niemiałam",

    # Móc / Chcieć
    "nie mogę": "niemogę", "nie możesz": "niemożesz", "nie może": "niemoże", "nie możemy": "niemożemy", "nie możecie": "niemożecie", "nie mogą": "niemogą", "nie mogłem": "niemogłem", "nie mogłam": "niemogłam",
    "nie chcę": "niechcę", "nie chce": "niechce", "nie chcesz": "niechcesz", "nie chcemy": "niechcemy", "nie chcecie": "niechcecie", "nie chcą": "niechcą", "nie chciał": "niechciał", "nie chciała": "niechciała",

    # Rozumieć / Lubić / Pamiętać
    "nie rozumiem": "nierozumiem", "nie rozumiesz": "nierozumiesz", "nie rozumie": "nierozumie",
    "nie lubię": "nielubię", "nie lubi": "nielubi", "nie lubisz": "nielubisz",
    "nie pamiętam": "niepamiętam", "nie pamiętasz": "niepamiętasz", "nie pamięta": "niepamięta",

    # Widzieć / Słyszeć
    "nie widzę": "niewidzę", "nie widzi": "niewidzi", "nie widział": "niewidział", "nie widziała": "niewidziała",
    "nie słyszę": "niesłyszę", "nie słyszy": "niesłyszy", "nie słyszał": "niesłyszał", "nie słyszała": "niesłyszała",

    # Działać / Być / Będzie
    "nie działa": "niedziała", "nie działają": "niedziałają", "nie działało": "niedziałało",
    "nie było": "niebyło", "nie był": "niebył", "nie była": "niebyła", "nie byli": "niebyli",
    "nie będzie": "niebędzie", "nie będą": "niebędą",

    # Robić / Zrobić
    "nie robi": "nierobi", "nie robią": "nierobią", "nie robił": "nierobił", "nie robiła": "nierobiła", "nie robić": "nierobić",
    "nie zrobisz": "niezrobisz", "nie zrobi": "niezrobi", "nie zrobił": "niezrobił", "nie zrobiła": "niezrobiła",

    # Dać / Pójść
    "nie da": "nieda", "nie dam": "niedam", "nie dasz": "niedasz", "nie dają": "niedają",
    "nie poszedł": "nieposzedł", "nie poszła": "nieposzła",

    # Wyrażenia predykatywne i bezokoliczniki
    "nie widać": "niewidać", "nie słychać": "niesłychać",
    "nie wolno": "niewolno", "nie warto": "niewarto", "nie trzeba": "nietrzeba", "nie można": "niemożna",

    # ── 6. Inne błędy i archaizmy składniowe ────────────────────────────
    "z wyjątkiem": "za wyjątkiem",
    "już": "jusz",

}

# Rule definitions: (name, regex pattern, replacement).

_RULE_DEFS: list[tuple[str, str, str]] = [
    # ── Nasal vowels ────────────────────────────────────────────────────
    ("ą→on",      r"ą(?=[tdcnszśźćkg])",  "on"),
    ("ą→om",      r"ą(?=[bpm])",         "om"),
    ("ą→om",      r"ą$",                 "om"),
    ("om→ą",      r"om$",                "ą"),
    ("ę→en",      r"ę(?=[tdcnszśźćkg])",  "en"),
    ("ę→em",      r"ę(?=[bpm])",         "em"),
    ("ę→e",       r"ę$",                 "e"),

    # ── Digraph confusions ──────────────────────────────────────────────
    ("ch→h",      r"ch",                 "h"),
    ("h→ch",      r"(?<!c)h",            "ch"),
    ("rz→ż",      r"rz",                 "ż"),
    ("ż→rz",      r"ż",                  "rz"),

    # ── ó / u ───────────────────────────────────────────────────────────
    ("ó→u",       r"ó",                 "u"),
    ("u→ó",       r"u(?=\w)",            "ó"),


    # ── Common suffix patterns ──────────────────────────────────────────
    ("ął→oł",    r"ął$",               "oł"),
    ("ęła→eła",  r"ęła$",              "eła"),
    ("ji→i",     r"ji$",               "i"),
    ("ii→i",     r"ii$",               "i"),
    ("ei→eji",     r"ei$",               "eji"),
    ("ai→aji",     r"ai$",               "aji"),
    ("oi→oji",     r"oi$",               "oji"),
    ("ść→źć",     r"ść$",                "źć"),
    ("źć→ść",     r"źć$",                "ść"),
    ("dź→ć",     r"dź$",                "ć"),

    # ── Consonant cluster simplifications ("Połykanie" liter) ───────────
    ("rwsz→rsz",  r"rwsz",               "rsz"),
    ("błk→bk",    r"błk",                "bk"),
    ("stn→sn",    r"stn",                "sn"), 
    ("szcz→scz",  r"szcz",               "scz"),

    # Consonant cluster reductions (rz -> sz, trz -> cz)
    ("prz→psz", r"prz", "psz"),
    ("krz→ksz", r"krz", "ksz"),
    ("grz→gsz", r"grz", "gsz"),
    ("trz→cz", r"trz", "cz")

]

_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (name, re.compile(pat, re.UNICODE), repl)
    for name, pat, repl in _RULE_DEFS
]

RULE_NAMES: frozenset[str] = frozenset(name for name, _, _ in _RULE_DEFS)


def _match_casing(
    template: str,
    replacement: str,
    *,
    container: str | None = None,
    start_index: int | None = None,
) -> str:
    source = container if (container is not None and start_index == 0) else template

    if source.isupper():
        return replacement.upper()
    if source.islower():
        return replacement.lower()

    return replacement.capitalize()


class SpellingErrorGenerator(ErrorGenerator):
    """Applies dictionary and rule-based spelling errors with a word budget."""

    def __init__(self, rate: float, seed: int) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be between 0.0 and 1.0, got {rate!r}")

        self.rate = rate
        self._seed = seed
        self._salt = "spelling"
        self._rules = list(_RULES)

        self._dictionary_patterns: list[tuple[re.Pattern[str], str]] = []

        for correct, incorrect in _SPELLING_DICT.items():
            pat = re.compile(
                rf"\b{re.escape(correct)}\b",
                re.IGNORECASE | re.UNICODE,
            )
            self._dictionary_patterns.append((pat, incorrect))


    def apply(self, text: str) -> str:
        rng_seed = deterministic_seed(self._seed, text, self._salt)
        rng = random.Random(rng_seed)

        target = int(self.rate * len(text.split(" ")) + 0.5)
        if target <= 0:
            return text

        # Phase 1: apply dictionary matches first.
        remaining = target
        for pattern, replacement in self._dictionary_patterns:
            if remaining <= 0:
                break
            text, used = self._replace_matches_budget(text, pattern, replacement, remaining, rng)
            remaining -= used

        if remaining <= 0:
            return text

        # Phase 2: apply rule-based edits on remaining words.
        return self._process_words_budget(text, remaining, rng)
    
    def _replace_matches_budget(
        self,
        text: str,
        pattern: re.Pattern[str],
        replacement: str,
        budget: int,
        rng: random.Random,
    ) -> tuple[str, int]:
        """Replace non-overlapping matches while consuming at most *budget* words."""
        
        matches = list(pattern.finditer(text))
        if not matches:
            return text, 0

        order = list(range(len(matches)))
        rng.shuffle(order)

        selected: set[int] = set()
        consumed = 0
        for idx in order:
            m = matches[idx]
            cost = 1
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
                parts.append(_match_casing(m.group(), replacement))
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

        # Collect all rule matches for this word.
        for _, pattern, repl in self._rules:
            for m in pattern.finditer(low):
                changes.append((m.start(), m.end(), repl))

        return changes

    def _process_words_budget(self, text: str, budget: int, rng: random.Random) -> str:
        """Apply up to *budget* individual non-overlapping changes."""
        tokens = re.findall(r"\w+|\W+", text, flags=re.UNICODE)
        if budget <= 0:
            return "".join(tokens)

        # token index -> candidate changes (start, end, replacement)
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
            adjusted = _match_casing(tok[start:end], repl, container=tok, start_index=start)
            tokens[token_idx] = tok[:start] + adjusted + tok[end:]
            budget -= 1

            # Keep non-overlapping candidates and shift trailing spans.
            delta = len(adjusted) - (end - start)
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

