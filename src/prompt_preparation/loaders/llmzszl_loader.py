from __future__ import annotations

import json
from pathlib import Path

from questions import LlmzszlQuestion, Question

from .base import Loader


class LLMZSZLLoader(Loader):
    def load(self, num_samples: int | None = None, seed: int = 42) -> list[Question]:
        """
        Load questions from a JSONL file, sampling 25% from each of:
        - Matura Math (type: Egzaminy Maturalne, name: Matematyka)
        - Matura Physics (type: Egzaminy Maturalne, name: Fizyka)
        - Matura Biology (type: Egzaminy Maturalne, name: Biologia)
        - Vocational Exams (type: Egzaminy Zawodowe, any name)
        Each category is sampled deterministically and then combined.
        """
        all_questions = {
            "math": [],
            "physics": [],
            "biology": [],
            "vocational": [],
        }

        with Path("datasets/llmzszl.jsonl").open("r", encoding="utf-8") as file:
            for line_no, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                typ = record.get("type")
                name = record.get("name")
                question_text = str(record.get("question", ""))
                answers = list(record.get("answers", []))
                correct_index = int(record.get("correct_answer_index", -1))
                q = LlmzszlQuestion(question_text, answers, correct_index)
                if typ == "Egzaminy Maturalne" and name == "Matematyka":
                    all_questions["math"].append(q)
                elif typ == "Egzaminy Maturalne" and name == "Fizyka":
                    all_questions["physics"].append(q)
                elif typ == "Egzaminy Maturalne" and name == "Biologia":
                    all_questions["biology"].append(q)
                elif typ == "Egzaminy Zawodowe":
                    all_questions["vocational"].append(q)

        # Determine how many samples per category
        if num_samples is None:
            # Use all available
            result = all_questions["math"] + all_questions["physics"] + all_questions["biology"] + all_questions["vocational"]
            return result

        # Proportions: math 25%, physics 25%, vocational 30%, biology 20%
        proportions = {"math": 0.25, "physics": 0.25, "vocational": 0.30, "biology": 0.20}
        cat_order = ["math", "physics", "vocational", "biology"]
        # Initial allocation (floor)
        per_cat_counts = {cat: int(num_samples * proportions[cat]) for cat in cat_order}
        # Distribute remainder deterministically
        allocated = sum(per_cat_counts.values())
        remainder = num_samples - allocated
        # Order for remainder: math, physics, vocational, biology
        for i in range(remainder):
            per_cat_counts[cat_order[i % len(cat_order)]] += 1

        sampled = []
        for cat in cat_order:
            qs = all_questions[cat]
            n = min(per_cat_counts[cat], len(qs))
            rng = __import__('random').Random(f"{seed}-{cat}")
            if n > 0:
                sampled.extend(rng.sample(qs, n))

        return sampled
