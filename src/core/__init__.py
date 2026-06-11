"""
Core models and neural network utilities for the DeepCFR Poker AI.
"""

from .model import PokerNetwork, encode_state, set_verbose

__all__ = [
    'PokerNetwork',
    'encode_state',
    'set_verbose',
]

# DeepCFRAgent requires the Rust `pokers` engine; keep it optional so that
# pure-Python environments (e.g. Ultimate Texas Hold'em) work without it.
try:
    from .deep_cfr import DeepCFRAgent  # noqa: F401
    __all__.append('DeepCFRAgent')
except ImportError:  # pragma: no cover - depends on pokers being installed
    DeepCFRAgent = None
