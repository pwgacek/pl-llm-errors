from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def build_llmzszl_judge_prompt(question: str, correct_answer: list[str], model_answer: str) -> str:
    return _load("llmzszl").format(
        question=question,
        correct_answer=", ".join(correct_answer),
        model_answer=model_answer,
    )


def build_polqa_judge_prompt(
    question: str, context: str, correct_answer: list[str], model_answer: str
) -> str:
    return _load("polqa").format(
        question=question,
        context=context,
        correct_answer=", ".join(correct_answer),
        model_answer=model_answer,
    )


def build_bbh_judge_prompt(
    question: str, correct_answer: list[list[str]], model_answer: str
) -> str:
    return _load("bbh").format(
        question=question,
        correct_answer=" | ".join(", ".join(order) for order in correct_answer),
        model_answer=model_answer,
    )
