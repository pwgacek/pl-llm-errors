from __future__ import annotations

import json
from pathlib import Path

from .base import Loader
from ..prompts import MaturaPrompt, Prompt


class MaturaLoader(Loader):
    def load(self, path: Path, num_samples: int, seed: int) -> list[Prompt]:
        prompts: list[Prompt] = []

        lines = self._load_lines(path)

        for line in lines:
            record = json.loads(line)

            task_text = str(record.get("tresc_zadania", "")).strip()
            key = str(record.get("klucz", "")).strip()
            points_raw = record.get("punkty_max", "")

            if not task_text or not key:
                raise ValueError(
                    "Invalid matura record, missing tresc_zadania or klucz."
                )

            points: int | str
            try:
                points = int(str(points_raw).strip())
            except ValueError:
                points = str(points_raw).strip()

            prompts.append(
                MaturaPrompt(
                    text=task_text,
                    key=key,
                    points=points,
                )
            )

        return prompts
