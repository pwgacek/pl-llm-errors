from .base import ErrorGenerator


class IdentityGenerator(ErrorGenerator):
    """Error generator that does nothing — returns the text unchanged."""

    def apply(self, text: str) -> str:
        return text
