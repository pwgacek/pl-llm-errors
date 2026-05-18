from .base import Loader
 
from .llmzszl_loader import LLMZSZLLoader
from .polqa_loader import PolQALoader
from .bbh_loader import BBHLoader
from .matematyka_rozszerzona_cke_loader import MatematykaRozszerzonaCKELoader

__all__ = [
	"Loader",
	"LLMZSZLLoader",
	"PolQALoader",
	"BBHLoader",
	"MatematykaRozszerzonaCKELoader",
]