from __future__ import annotations

from .cv2 import Cv2Backend
from .protocol import Backend
from .pystackreg import PystackregBackend
from .scikit import ScikitBackend


def get_backend(name: str = "pystackreg") -> Backend:
    """Return a backend instance by name."""
    normalized = name.lower()
    if normalized == "pystackreg":
        return PystackregBackend()
    if normalized == "scikit":
        return ScikitBackend()
    if normalized == "cv2":
        return Cv2Backend()
    raise ValueError(
        f"Unsupported backend '{name}'. Available backends: 'pystackreg', 'scikit', 'cv2'."
    )


__all__ = [
    "PystackregBackend",
    "ScikitBackend",
    "Cv2Backend",
    "Backend",
    "get_backend",
]
