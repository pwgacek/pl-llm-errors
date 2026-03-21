
import json
import re
from pathlib import Path

from .base import Loader
from questions import BBHQuestion, Question

OPTION_PATTERN = re.compile(r"^\(([A-G])\)\s*(.+)$")


def _parse_bbh_input(raw: str) -> tuple[str, list[str]] | None:
    """Split input into (question_text, [option_a, ..., option_g])."""
    # Split on "Opcje:" to separate narrative from options
    parts = raw.split("Opcje:")
    if len(parts) != 2:
        return None
    
    question_text = parts[0].strip()
    options_text = parts[1].strip()
    
    lines = [line.strip() for line in options_text.splitlines() if line.strip()]
    answers: dict[str, str] = {}
    
    for line in lines:
        m = OPTION_PATTERN.match(line)
        if m:
            answers[m.group(1)] = m.group(2).strip()
    
    if not question_text or not answers:
        return None
    
    ordered = [answers[letter] for letter in "ABCDEFG" if letter in answers]
    return question_text, ordered


class BBHLoader(Loader):
    def load(self, num_samples: int | None = None, seed: int = 42) -> list[Question]:
        """Load questions from the BBH logical deduction dataset JSONL file, optionally sampling deterministically."""
        questions: list[Question] = []

        with Path("datasets/bbh-logical-deduction-seven-objects-pl.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                input_text = record.get("input", "").strip()
                answer = record.get("target", "").strip().strip("()")

                if not input_text or not answer:
                    continue
                
                parsed = _parse_bbh_input(input_text)
                if parsed is None:
                    continue
                
                question_text, options = parsed
                questions.append(BBHQuestion(question_text, options, answer))

        return self._deterministic_sample(questions, num_samples, seed)