
import json
import re
from pathlib import Path

from .base import Loader
from ..questions import BBHQuestion, Question

OPTION_PATTERN = re.compile(r"^\(([A-G])\)\s*(.+)$")


def _parse_bbh_input(raw: str) -> tuple[str, list[str]]:
    """Split input into (question_text, [option_a, ..., option_g])."""

    parts = raw.split("Opcje:")
    if len(parts) != 2:
        raise ValueError(f"Unexpected BBH input format, missing 'Opcje:' separator: {raw}")
    
    question_text = parts[0].strip()
    options_text = parts[1].strip()
    
    lines = [line.strip() for line in options_text.splitlines() if line.strip()]
    answers: list[str] = []
    
    for line in lines:
        m = OPTION_PATTERN.match(line)
        if m:
            answers.append(m.group(2).strip())
    
    if not question_text or not len(answers) == 7:
        raise ValueError(f"Unexpected BBH input format, missing question text or options: {raw}")

    return question_text, answers


class BBHLoader(Loader):
    def load(self, path: Path, num_samples: int, seed: int) -> list[Question]:
        questions: list[Question] = []

        lines = self._load_lines(path)
        lines = self._deterministic_sample(lines, num_samples, seed)

        for line in lines:
            record = json.loads(line)
            input_text = record.get("input", "").strip()
            answer = record.get("target", "").strip().strip("()")

            if not input_text or not answer:
                raise ValueError(f"Invalid BBH record, missing input or target: {line}")
            
            question_text, options = _parse_bbh_input(input_text)
            questions.append(BBHQuestion(question_text, options, answer))

        return questions