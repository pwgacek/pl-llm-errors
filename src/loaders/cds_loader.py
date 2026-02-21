from __future__ import annotations

import csv
from pathlib import Path

from questions import TextualEntailmentQuestion, Question

from .base import Loader


class CDSLoader(Loader):
    def load(self) -> list[Question]:
        """Load textual entailment questions from CDS_test.csv file."""
        questions: list[Question] = []

        with Path("datasets/CDS_test.csv").open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                sentence_a = row.get("sentence_A", "").strip()
                sentence_b = row.get("sentence_B", "").strip()
                entailment_judgment = row.get("entailment_judgment", "").strip()
                
                if sentence_a and sentence_b and entailment_judgment:
                    questions.append(TextualEntailmentQuestion(sentence_a, sentence_b, entailment_judgment))

        return questions