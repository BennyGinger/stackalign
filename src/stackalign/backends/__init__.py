from __future__ import annotations

from .protocol import Backend
from .pystackreg_backend import PystackregBackend


def get_backend(name: str = "pystackreg") -> Backend:
    """Return a backend instance by name."""
    normalized = name.lower()
    if normalized != "pystackreg":
        raise ValueError(
            f"Unsupported backend '{name}'. Only 'pystackreg' is available in this version."
        )
    return PystackregBackend()


__all__ = [
    "PystackregBackend",
    "Backend",
    "get_backend",
]
