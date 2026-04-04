import re
import spacy

from .base import ErrorGenerator

_INNER_PUNCTUATION_CHARS = (
    ",;:"
    "\u2013\u2014" # pauza i półpauza
    "\"'" # proste cudzysłowy
    "\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u00ab\u00bb" # polskie cudzysłowy
    "()[]{}<>" # nawiasy
    "\u2026" # wielokropek
)

_ALL_PUNCTUATION_RE = re.compile(f"[{re.escape('.!?' + _INNER_PUNCTUATION_CHARS)}]")
_INNER_PUNCTUATION_RE = re.compile(f"[{re.escape(_INNER_PUNCTUATION_CHARS)}]")

# --- NEW: protection regexes ---
_NUMBER_RE = re.compile(r"\b\d+[.,:]\d+\b")
_NUMBER_RANGE_DASH_RE = re.compile(r"\b\d+[\u2013\u2014]\d+\b")
_ORDINAL_POSITION_RE = re.compile(r"\b\d+\.(?=\s+[a-ząćęłńóśźż])")
_DATE_RE = re.compile(r"\b\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\b")
_EMAIL_RE = re.compile(r"\b\S+@\S+\b")
_URL_RE = re.compile(r"\b\w+\.\w+\b")

_PROTECTION_PATTERNS = [
    _NUMBER_RE,
    _NUMBER_RANGE_DASH_RE,
    _ORDINAL_POSITION_RE,
    _DATE_RE,
    _EMAIL_RE,
    _URL_RE,
]

# --- NLP ---
_PRESERVE_ENTITY_TYPES = {
    "PERSON", "GPE", "ORG", "PRODUCT",
    "LOC", "EVENT", "WORK_OF_ART", "LANGUAGE", "NORP"
}
_ACRONYM_RE = re.compile(r"^(?=.*[A-Z]{2,})[A-Z0-9-]+$")
_NLP_MODEL = spacy.load("pl_core_news_sm")

_COMMON_ABBREVIATIONS = {
    "a.",         # albo
    "al.",        # aleja
    "ang.",       # angielski
    "ar.",        # arabski
    "arab.",      # arabski
    "arch.",      # architekt
    "b.",         # były (np. b. wicepremier)
    "bł.",        # błogosławiony
    "cdn.",       # ciąg dalszy nastąpi
    "doc.",       # docent
    "dosł.",      # dosłownie
    "dr.",        # doktora/doktorowi (uwaga: w mianowniku bez kropki)
    "ds.",        # do spraw
    "gen.",       # generał
    "godz.",      # godzina
    "gr.",        # grecki
    "grc.",       # grecki (starogrecki/koine)
    "hab.",       # habilitowany (np. dr hab.)
    "haw.",       # hawajski
    "hiszp.",     # hiszpański
    "hol.",       # holenderski
    "im.",        # imienia
    "inż.",       # inżynier
    "itd.",       # i tak dalej
    "itp.",       # i tym podobne
    "jw.",        # jak wyżej
    "katal.",     # kataloński
    "kom.",       # komórkowy (np. tel. kom.)
    "lek.",       # lekarz
    "łac.",       # łaciński / łacina
    "m.in.",      # między innymi
    "m.st.",      # miasta stołecznego
    "mgr.",       # magistra/magistrowi (podobnie jak dr - w mianowniku bez kropki)
    "min.",       # minuta
    "n.e.",       # naszej ery
    "niem.",      # niemiecki
    "np.",        # na przykład
    "op.",        # opus (dzieło muzyczne)
    "p.n.e.",     # przed naszą erą
    "p.o.",       # pełniący obowiązki
    "pl.",        # plac / polski
    "por.",       # porównaj
    "pow.",       # powiat
    "poz.",       # pozycja (w dzienniku ustaw)
    "prof.",      # profesor
    "ps.",        # pseudonim
    "pt.",        # pod tytułem
    "r.",         # rok
    "red.",       # redakcja / redaktor
    "scs.",       # starocerkiewnosłowiański
    "sek.",       # sekunda
    "sp.",        # spółka
    "St.",        # Saint / Sankt (z ang./niem. święty, np. St. Moritz)
    "starogr.",   # starogrecki
    "str.",       # strona
    "św.",        # święty
    "tel.",       # telefon
    "tłum.",      # tłumaczenie / tłumacz
    "tr.",        # turecki
    "tys.",       # tysiąc
    "tzn.",       # to znaczy
    "tzw.",       # tak zwany
    "ul.",        # ulica
    "ur.",        # urodzony
    "w.",         # wiek
    "wł.",        # włoski
    "właśc.",     # właściwie
    "woj.",       # województwo
    "wyd.",       # wydanie
    "wym.",       # wymowa
    "z o.o.",     # z ograniczoną odpowiedzialnością
    "zm.",        # zmarły
    "zob.",       # zobacz
    "zw."         # zwany
}


def _is_acronym_token(text: str) -> bool:
    return bool(_ACRONYM_RE.fullmatch(text))


def _should_preserve(token) -> bool:
    if _is_acronym_token(token.text):
        return True
    if token.pos_ == "PROPN":
        return True
    if token.ent_type_ in _PRESERVE_ENTITY_TYPES:
        return True
    return False


def _lowercase_sentence_starts(text: str) -> str:
    doc = _NLP_MODEL(text)
    result = []

    for token in doc:
        if token.is_sent_start:
            if _should_preserve(token):
                result.append(token.text)
            else:
                result.append(token.text.lower())
        else:
            result.append(token.text)

        result.append(token.whitespace_)

    return "".join(result)


# --- NEW: protection helpers ---
def _protect_spans(text: str):
    matches = []

    for pattern in _PROTECTION_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), m.group()))

    # sort and remove overlaps
    matches.sort()
    filtered = []
    last_end = -1
    for start, end, val in matches:
        if start >= last_end:
            filtered.append((start, end, val))
            last_end = end

    result = []
    placeholders = []
    last = 0

    for i, (start, end, value) in enumerate(filtered):
        result.append(text[last:start])
        placeholder = f"__P{i}__"
        result.append(placeholder)
        placeholders.append((placeholder, value))
        last = end

    result.append(text[last:])
    return "".join(result), placeholders


def _restore_spans(text: str, placeholders):
    for placeholder, value in placeholders:
        text = text.replace(placeholder, value)
    return text


def _remove_all_punctuation(text: str) -> str:
    return _ALL_PUNCTUATION_RE.sub("", text)


# --- NEW: abbreviation protection ---
def _protect_abbreviations(text: str):
    placeholders = []
    result = text

    for abbr in _COMMON_ABBREVIATIONS:
        pattern = re.compile(rf"\b{re.escape(abbr)}", re.IGNORECASE)

        def repl(match):
            placeholder = f"__ABBR_{len(placeholders)}__"
            placeholders.append((placeholder, match.group()))
            return placeholder

        result = pattern.sub(repl, result)

    return result, placeholders


def _restore_abbreviations(text: str, placeholders: list[tuple[str, str]]) -> str:
    placeholder_to_original = dict(placeholders)

    def repl(match: re.Match[str]) -> str:
        placeholder = match.group(0)
        original = placeholder_to_original.get(placeholder, placeholder)

        # Abbreviations are normalized to lowercase in degraded punctuation mode.
        return original.lower()

    return re.sub(r"__ABBR_\d+__", repl, text)


class PunctuationAllErrorGenerator(ErrorGenerator):
    """Removes all punctuation and lowercases sentence-start words (safe for numbers, dates, abbreviations)."""

    def apply(self, text: str) -> str:
        text = _lowercase_sentence_starts(text)

        # Keep sentence boundaries from abbreviation dots during sentence-start lowering.
        text, abbr_placeholders = _protect_abbreviations(text)
        text, other_placeholders = _protect_spans(text)
        text = _remove_all_punctuation(text)

        text = _restore_spans(text, other_placeholders)
        text = _restore_abbreviations(text, abbr_placeholders)

        return text


class PunctuationInnerErrorGenerator(ErrorGenerator):
    """Removes only inner punctuation (safe for numbers, dates, etc.)."""

    def apply(self, text: str) -> str:
        text, placeholders = _protect_spans(text)
        text = _INNER_PUNCTUATION_RE.sub("", text)

        return _restore_spans(text, placeholders)