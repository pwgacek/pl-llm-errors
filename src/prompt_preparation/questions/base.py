from __future__ import annotations

from abc import ABC, abstractmethod

from errors.base import ErrorGenerator

class Question(ABC):
    @abstractmethod
    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        raise NotImplementedError
