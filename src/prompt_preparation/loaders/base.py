from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import json
from typing import Any
from questions import Question
from pathlib import Path

class Loader(ABC):
    @staticmethod
    def _deterministic_sample(data: list[Any], num_samples: int, seed: int) -> list[Question]:
        """Deterministically sample num_samples from questions using the given seed."""
        if num_samples > len(data):
            raise ValueError(f"num_samples={num_samples} exceeds available data size {len(data)}")
        
        rng = __import__('random').Random(seed)
        return rng.sample(data, num_samples)

    
    @staticmethod
    def _load_lines(path: Path) -> list[str]:
        """Load raw lines from the dataset file."""
        with path.open("r", encoding="utf-8") as file:
            return [line for line in file]
        
    @staticmethod
    def _load_rows(path: Path, delimiter: str = ',') -> list:
        """Load rows from the dataset file."""
        with path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            return [row for row in reader]

    @staticmethod
    def _load_json(path: Path, delimiter: str = ',') -> Any:
        """Load records from the dataset file."""
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    
    @abstractmethod
    def load(self, path: Path, num_samples: int, seed: int) -> list[Question]:
        """
        Load up to num_samples questions, deterministically if seed is set.
        If num_samples is None, load all questions.
        """
        raise NotImplementedError
