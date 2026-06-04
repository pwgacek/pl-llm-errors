from __future__ import annotations

import json
from pathlib import Path

from .base import Loader
from ..prompts import IFEvalPrompt, Prompt


class IFEvalLoader(Loader):
    def load(self, path: Path, num_samples: int, seed: int) -> list[Prompt]:
        prompts: list[Prompt] = []

        lines = self._load_lines(path)
        for line in lines:
            record = json.loads(line)

            prompt_text = str(record.get("prompt", "")).strip()
            if not prompt_text:
                raise ValueError("Invalid IFEval record, missing prompt text.")

            instruction_id_list = record.get("instruction_id_list", [])
            if not isinstance(instruction_id_list, list):
                raise ValueError("Invalid IFEval record, instruction_id_list must be a list.")
            instruction_ids = [str(item) for item in instruction_id_list]

            kwargs_list = record.get("kwargs", [])
            if not isinstance(kwargs_list, list):
                raise ValueError("Invalid IFEval record, kwargs must be a list.")


            prompts.append(
                IFEvalPrompt(
                    prompt=prompt_text,
                    instruction_id_list=instruction_ids,
                    kwargs=kwargs_list,
                )
            )

        return prompts
