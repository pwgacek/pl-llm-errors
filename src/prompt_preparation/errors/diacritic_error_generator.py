from .base import ErrorGenerator

_POLISH_MAP = str.maketrans(
    "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
    "acelnoszzACELNOSZZ",
)


class DiacriticErrorGenerator(ErrorGenerator):
    """Replaces Polish diacritic letters with their ASCII equivalents (e.g. ł->l, ą->a)."""

    def apply(self, text: str) -> str:
        return text.translate(_POLISH_MAP)
