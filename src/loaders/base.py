from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from questions import Question


class Loader(ABC):
    @abstractmethod
    def load(self) -> list[Question]:
        raise NotImplementedError
