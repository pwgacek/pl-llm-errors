from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class VerificationResult(Enum):
    CORRECT = "Correct"
    INCORRECT = "Incorrect"
    ERROR = "Error"


class Question(ABC):
    @abstractmethod
    def verify_answer(self, llm_answer: str) -> VerificationResult:
        raise NotImplementedError
    
    @abstractmethod
    def build_prompt(self) -> str:
        raise NotImplementedError
