import re
from pathlib import Path
from ..questions import LDEKQuestion, Question
from .base import Loader

OPTION_PATTERN = re.compile(r"^([A-E])\.\s*(.+)$")


def _parse_question_w_options(raw: str) -> tuple[str, list[str]] | None:
    lines = [line.strip() for line in raw.strip().splitlines()]

    question_lines: list[str] = []
    answers: list[str] = []

    for line in lines:
        m = OPTION_PATTERN.match(line)
        if m:
            answers.append(m.group(2).strip())
        else:
            question_lines.append(line)

    if not question_lines or not answers:
        return None
    
    return " ".join(question_lines).strip(), answers


class LDEKLoader(Loader):
    def load(self, path: Path, num_samples: int, seed: int) -> list[Question]:
        questions: list[Question] = []

        records = self._load_json(path)

        for record in records:
            raw = record.get("question_w_options", "")
            answer_letter = record.get("answer", "").strip().upper()

            if not raw or answer_letter not in "ABCDE":
                raise ValueError(f"Invalid LDEK record, missing question_w_options or invalid answer letter: {record}")

            parsed = _parse_question_w_options(raw)
            if not parsed: # some records are not parsable so we skip them
                continue

            question_text, answers = parsed

            questions.append(LDEKQuestion(question_text, answers, answer_letter))

        return self._deterministic_sample(questions, num_samples, seed)
