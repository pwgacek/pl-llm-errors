from __future__ import annotations

from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load(name: str) -> str:
    return (_TEMPLATES_DIR / f"{name}.txt").read_text(encoding="utf-8")


def _render(name: str, values: dict[str, str]) -> str:
    template = _load(name)
    for key, value in values.items():
        template = template.replace(f"{{{key}}}", value)
    return template




def build_bbh_judge_prompt(
    question: str, correct_answer: list[list[str]], model_answer: str
) -> str:
    return _render(
        "bbh",
        {
            "question": question,
            "correct_answer": " | ".join(", ".join(order) for order in correct_answer),
            "model_answer": model_answer,
        },
    )


def build_matura_judge_prompt(
    task: str, key: str, points: int | str, model_answer: str
) -> str:
    return _render(
        "matura",
        {
            "task": task,
            "key": key,
            "points": str(points),
            "model_answer": model_answer,
        },
    )
