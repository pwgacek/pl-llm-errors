from .base import Prompt
from .llmzszl_prompt import LlmzszlPrompt
from .polqa_prompt import PolQAPrompt
from .bbh_prompt import BBHPrompt
from .matematyka_rozszerzona_cke_prompt import MatematykaRozszerzonaCKEPrompt
from .ifeval_prompt import IFEvalPrompt

__all__ = [
    "Prompt",
    "LlmzszlPrompt",
    "PolQAPrompt",
    "BBHPrompt",
    "MatematykaRozszerzonaCKEPrompt",
    "IFEvalPrompt",
]
