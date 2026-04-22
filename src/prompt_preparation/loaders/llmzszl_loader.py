import json
from ..prompts import LlmzszlPrompt, Prompt
from .base import Loader
from pathlib import Path


class LLMZSZLLoader(Loader):
    def load(self, path: Path, num_samples: int, seed: int) -> list[Prompt]:
        prompts: list[Prompt] = []

        lines = self._load_lines(path)

        for line in lines:
            record = json.loads(line)

            question_text = str(record.get("question", ""))
            answer = str(record.get("answer", ""))

            if not question_text or not answer:
                raise ValueError(f"Invalid LLMZSZL record, missing question or answer: {line}")

            prompts.append(LlmzszlPrompt(question_text, answer))

        return self._deterministic_sample(prompts, num_samples, seed)
