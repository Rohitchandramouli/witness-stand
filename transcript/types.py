"""Shared transcript type aliases.

The canonical definitions live in models.py.
This module re-exports them so transcript users can import from transcript.types.
"""

from models import Speaker, Turn, TurnType

__all__ = [
    "Turn",
    "Speaker",
    "TurnType",
]