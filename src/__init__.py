"""Top-level package for the USDA QA bot source modules."""

__version__ = "2.2.0"

from . import dataset, training, inference, utils
try:
    from . import rag
except (ImportError, RuntimeError):
    rag = None
try:
    from . import Hybrid_RAG
except (ImportError, RuntimeError):
    Hybrid_RAG = None

__all__ = [
    "dataset",
    "training",
    "inference",
    "rag",
    "Hybrid_RAG",
    "utils",
]
