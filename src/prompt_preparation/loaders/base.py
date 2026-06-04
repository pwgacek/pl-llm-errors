from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import json
from typing import Any
from ..prompts import Prompt
from pathlib import Path

class Loader(ABC):
    
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
    def _load_json(path: Path) -> Any:
        """Load records from the dataset file."""
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    
    @abstractmethod
    def load(self, path: Path) -> list[Prompt]:
        """
        Load prompts.
        """
        raise NotImplementedError
