"""Micro-Transformer: a small decoder-only Transformer built from scratch in NumPy."""
from .model import Config, MicroTransformer
from .tokenizer import BPETokenizer

__all__ = ["Config", "MicroTransformer", "BPETokenizer"]
__version__ = "1.0.0"
