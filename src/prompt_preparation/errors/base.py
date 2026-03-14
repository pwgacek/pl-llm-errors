from abc import ABC, abstractmethod


class ErrorGenerator(ABC):
    """Base class for all error generators that introduce errors into text."""

    @abstractmethod
    def apply(self, text: str) -> str:
        """Apply the error to the given text and return the modified string."""
        ...
