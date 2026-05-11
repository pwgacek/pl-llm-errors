from .base import Prompt
from .llmzszl_prompt import LlmzszlPrompt
from .llmzszl_mc_prompt import LlmzszlMCPrompt
from .polqa_prompt import PolQAPrompt
from .bbh_prompt import BBHPrompt

__all__ = [
    "Prompt",
    "LlmzszlPrompt",
    "LlmzszlMCPrompt",
    "PolQAPrompt",
    "BBHPrompt",
]
