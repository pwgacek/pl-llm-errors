from __future__ import annotations

from abc import ABC, abstractmethod
from questions import Question


class Loader(ABC):
    @staticmethod
    def _deterministic_sample(questions: list[Question], num_samples: int | None, seed: int = 42) -> list[Question]:
        """Deterministically sample num_samples from questions using the given seed."""
        if num_samples is not None and num_samples < len(questions):
            rng = __import__('random').Random(seed)
            return rng.sample(questions, num_samples)
        return questions
    
    @abstractmethod
    def load(self, num_samples: int | None = None, seed: int = 42) -> list[Question]:
        """
        Load up to num_samples questions, deterministically if seed is set.
        If num_samples is None, load all questions.
        """
        raise NotImplementedError
