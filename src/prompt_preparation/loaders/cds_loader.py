from questions import CdsQuestion, Question
from .base import Loader
from pathlib import Path


class CDSLoader(Loader):
    def load(self, path: Path, num_samples: int, seed: int) -> list[Question]:
        questions: list[Question] = []

        rows = self._load_rows(path, delimiter='\t')

        for row in rows:
            sentence_a = row.get("sentence_A", "").strip()
            sentence_b = row.get("sentence_B", "").strip()
            entailment_judgment = row.get("entailment_judgment", "").strip()

            if not sentence_a or not sentence_b or not entailment_judgment:
                raise ValueError(f"Invalid CDS record, missing sentence_A, sentence_B, or entailment_judgment: {row}")
            
            questions.append(CdsQuestion(sentence_a, sentence_b, entailment_judgment))

        return self._deterministic_sample(questions, num_samples, seed)