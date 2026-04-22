from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..errors.base import ErrorGenerator

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_template(name: str) -> str:
    return (_TEMPLATES_DIR / f"{name}.txt").read_text(encoding="utf-8")


def render_template(name: str, values: dict[str, str]) -> str:
    template = _load_template(name)
    for key, value in values.items():
        template = template.replace(f"{{{key}}}", value)
    return template


class Prompt(ABC):
    @staticmethod
    def _format_lettered_choices(options: list[str]) -> str:
        return "\n".join(
            f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(options)
        )

    @abstractmethod
    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        raise NotImplementedError