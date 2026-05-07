from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def shift_to_tmat(shift_yx: NDArray[np.floating] | tuple[float, float]) -> NDArray[np.float64]:
    """Convert a (shift_y, shift_x) pair to a 3x3 homogeneous translation matrix.

    ``phase_cross_correlation`` returns shifts in ``(y, x)`` order. stackalign
    stores translations in homogeneous matrices in ``(x, y)`` convention,
    i.e. ``tmat[0, 2]`` is x-translation and ``tmat[1, 2]`` is y-translation.
    scipy ``ndimage.shift`` consumes shifts in ``(y, x)`` order, so callers
    must decode these matrix entries back to ``(y, x)`` before applying.

    The matrix stored here is::

        [[1, 0, shift_x],
         [0, 1, shift_y],
         [0, 0,       1]]
    """
    tmat = np.eye(3, dtype=np.float64)
    tmat[0, 2] = float(shift_yx[1])  # shift_x
    tmat[1, 2] = float(shift_yx[0])  # shift_y
    return tmat


_SCIKIT_SUPPORTED_METHODS: frozenset[str] = frozenset({"translation"})


def validate_method(method: str) -> None:
    if method not in _SCIKIT_SUPPORTED_METHODS:
        raise ValueError(f"scikit backend only supports method='translation'. Got '{method}'.")
