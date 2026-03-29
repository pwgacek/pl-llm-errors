from .base import Loader
 
from .cds_loader import CDSLoader
from .ldek_loader import LDEKLoader
from .llmzszl_loader import LLMZSZLLoader
from .polqa_loader import PolQALoader
from .bbh_loader import BBHLoader
from .bbh_open_loader import BBHOpenLoader

__all__ = [
	"Loader",
	"CDSLoader",
	"LDEKLoader",
	"LLMZSZLLoader",
	"PolQALoader",
	"BBHLoader",
	"BBHOpenLoader",
]