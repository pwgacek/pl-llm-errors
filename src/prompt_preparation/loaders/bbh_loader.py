
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
    def __init__(self, answer_permutation: list[int] | None = None) -> None:
        self.answer_permutation = answer_permutation

    @staticmethod
    def _validate_permutation(permutation: list[int], num_answers: int) -> None:
        expected = list(range(num_answers))
        if sorted(permutation) != expected:
            raise ValueError(
                f"Invalid BBH answer_permutation={permutation}, expected permutation of {expected}"
            )

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
            correct_index = ord(answer.upper()) - ord("A")
            if correct_index < 0 or correct_index >= len(options):
                raise ValueError(f"Invalid BBH target answer '{answer}' for options: {line}")

            if self.answer_permutation is not None:
                self._validate_permutation(self.answer_permutation, len(options))
                options = [options[idx] for idx in self.answer_permutation]
                correct_index = self.answer_permutation.index(correct_index)

            answer = chr(ord("A") + correct_index)
            questions.append(BBHQuestion(question_text, options, answer))

        return questions