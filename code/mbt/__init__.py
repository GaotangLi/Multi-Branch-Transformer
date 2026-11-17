"""Explicit Multi-branch Transformer models and native MoE pretraining."""

from .config import ModelConfig
from .model import build_model, make_native_config

__all__ = ["ModelConfig", "build_model", "make_native_config", "MBTModel", "MBTDecoder"]
__version__ = "0.2.0"


def __getattr__(name):
    # Configuration/CLI use needs no PyTorch or GPU runtime. Load those
    # dependencies only when the actual model classes are requested.
    if name == "MBTModel":
        from .model import _prepare_backend
        _prepare_backend()
        from .modeling import MBTModel
        return MBTModel
    if name == "MBTDecoder":
        from .decoder import MBTDecoder
        return MBTDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
