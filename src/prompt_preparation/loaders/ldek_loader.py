from __future__ import annotations

import json
import re
from pathlib import Path

from questions import LDEKQuestion, Question

from .base import Loader

OPTION_PATTERN = re.compile(r"^([A-E])\.\s*(.+)$")


def _parse_question_w_options(raw: str) -> tuple[str, list[str]] | None:
    """Split a question_w_options string into (question_text, [answer_a, ..., answer_e])."""
    lines = [line.strip() for line in raw.strip().splitlines()]

    question_lines: list[str] = []
    answers: dict[str, str] = {}

    for line in lines:
        m = OPTION_PATTERN.match(line)
        if m:
            answers[m.group(1)] = m.group(2).strip()
        else:
            if not answers:  # still in question text part
                question_lines.append(line)

    if not question_lines or not answers:
        return None

    ordered = [answers[letter] for letter in "ABCDE" if letter in answers]
    return " ".join(question_lines).strip(), ordered


class LDEKLoader(Loader):
    def load(self) -> list[Question]:
        """Load multiple-choice questions from the LDEK medical exam JSON file."""
        questions: list[Question] = []

        with Path("datasets/medical-exams-LDEK-PL-2008-2024.json").open("r", encoding="utf-8") as file:
            records = json.load(file)

        for record in records:
            raw = record.get("question_w_options", "")
            answer_letter = record.get("answer", "").strip().upper()

            if not raw or answer_letter not in "ABCDE":
                continue

            parsed = _parse_question_w_options(raw)
            if parsed is None:
                continue

            question_text, answers = parsed
            questions.append(LDEKQuestion(question_text, answers, answer_letter))

        return questions
