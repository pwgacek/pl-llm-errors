from __future__ import annotations

from abc import ABC, abstractmethod

from ..errors.base import ErrorGenerator


class Question(ABC):
    @staticmethod
    def _format_lettered_choices(options: list[str]) -> str:
        return "\n".join(
            f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(options)
        )

    @abstractmethod
    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        raise NotImplementedError
