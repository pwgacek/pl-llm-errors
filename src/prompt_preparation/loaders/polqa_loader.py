from __future__ import annotations
import ast
import math
from ..prompts import PolQAPrompt, Prompt
from .base import Loader
from pathlib import Path

class PolQALoader(Loader):

    def __init__(self) -> None:
        self.long_context_ratio = 0.1

    @staticmethod
    def _parse_answers(raw: str) -> list[str]:
        parsed = ast.literal_eval(raw)
        return [str(a) for a in parsed]

    @staticmethod
    def _take_longest_contexts(
        prompts: list[PolQAPrompt],
        long_context_ratio: float,
    ) -> list[PolQAPrompt]:
        if not 0 < long_context_ratio <= 1:
            raise ValueError(
                f"long_context_ratio={long_context_ratio} must be in (0, 1]."
            )

        if not prompts:
            return prompts

        keep_count = max(1, math.ceil(len(prompts) * long_context_ratio))
        sorted_prompts = sorted(prompts, key=lambda p: len(p.context), reverse=True)
        return sorted_prompts[:keep_count]
 
        

    def load(self, path: Path, num_samples: int, seed: int) -> list[Prompt]:
        prompts: list[PolQAPrompt] = []
        seen: set[tuple[str, str]] = set()

        rows = self._load_rows(path)
        for row in rows:
            if row.get("relevant", "").strip() == "True":
                question_text = row.get("question", "").strip()
                context = row.get("passage_text", "").strip()
                answers = self._parse_answers(row.get("answers", "[]"))

                if not question_text or not context or not answers:
                    raise ValueError(f"Invalid POLQA record, missing question text or passage text: {row}")

                # handle duplicates as duplicated fields does not always is set correctly
                key = (question_text, context)
                if key in seen:
                    continue
                seen.add(key)

                prompts.append(PolQAPrompt(question_text, context, answers))

        if self.long_context_ratio is not None:
            prompts = self._take_longest_contexts(prompts, self.long_context_ratio)

        return self._deterministic_sample(prompts, num_samples, seed)