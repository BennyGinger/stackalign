from __future__ import annotations

from .protocol import Backend


def get_backend(name: str = "pystackreg") -> Backend:
    """
    Return a backend instance by name.
    """
    normalized = name.lower()
    if normalized == "pystackreg":
        from .pystackreg import PystackregBackend
        return PystackregBackend()
    if normalized == "scikit":
        from .scikit import ScikitBackend
        return ScikitBackend()
    if normalized == "cv2":
        from .cv2 import Cv2Backend
        return Cv2Backend()
    raise ValueError(f"Unsupported backend '{name}'. Available backends: 'pystackreg', 'scikit', 'cv2'.")


__all__ = ["Backend",
    "get_backend",
]
