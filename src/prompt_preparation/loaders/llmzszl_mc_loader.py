import json
from pathlib import Path

from ..prompts import LlmzszlMCPrompt, Prompt
from .base import Loader


class LLMZSZLMCLoader(Loader):
    def load(self, path: Path, num_samples: int, seed: int) -> list[Prompt]:
        prompts: list[Prompt] = []

        lines = self._load_lines(path)

        for line in lines:
            record = json.loads(line)

            question_text = str(record.get("question", ""))
            answers = record.get("answers", [])
            correct_idx = record.get("correct_answer_index", None)

            if not question_text or not answers:
                raise ValueError(f"Invalid LLMZSZL MC record, missing question or answers: {line}")

            # ensure answers are strings
            answers = [str(a) for a in answers]

            prompts.append(LlmzszlMCPrompt(question_text, answers, correct_idx))

        return self._deterministic_sample(prompts, num_samples, seed)
