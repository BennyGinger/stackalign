from __future__ import annotations

from typing import cast, get_args

import numpy as np
from numpy.typing import NDArray
from pystackreg import StackReg

from stackalign.constants import Method, ReferenceStrategy

_METHOD_TO_STACKREG = {
    "translation": StackReg.TRANSLATION,
    "rigid_body": StackReg.RIGID_BODY,
    "affine": StackReg.AFFINE,
}

_REFERENCE_STRATEGIES = cast(tuple[ReferenceStrategy, ...], get_args(ReferenceStrategy))
_VALID_REFERENCE_STRATEGIES: frozenset[str] = frozenset(_REFERENCE_STRATEGIES)


def validate_method(method: str) -> None:
    if method not in _METHOD_TO_STACKREG:
        raise ValueError(f"Unsupported method='{method}'. Supported methods: {sorted(_METHOD_TO_STACKREG)}")


def validate_reference_strategy(reference_strategy: str) -> None:
    if reference_strategy not in _VALID_REFERENCE_STRATEGIES:
        raise ValueError(f"Unsupported reference_strategy='{reference_strategy}'. Supported: {sorted(_VALID_REFERENCE_STRATEGIES)}")


def create_stackreg(method: Method) -> StackReg:
    return StackReg(_METHOD_TO_STACKREG[method])


def identity_tmats(length: int) -> NDArray[np.float64]:
    """Create a (length, 3, 3) array of identity transformation matrices."""
    if length < 0:
        raise ValueError(f"length must be >= 0. Got {length}.")
    return np.repeat(np.eye(3, dtype=np.float64)[None, :, :], length, axis=0)


def accumulate_pairwise_tmats(pairwise_tmats: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert pairwise (t->t-1) tmats into cumulative tmats in frame-0 coordinates."""
    if pairwise_tmats.ndim != 3 or pairwise_tmats.shape[1:] != (3, 3):
        raise ValueError(f"pairwise_tmats must have shape (T, 3, 3). Got {pairwise_tmats.shape}.")

    cumulative = identity_tmats(pairwise_tmats.shape[0])
    for frame_index in range(1, pairwise_tmats.shape[0]):
        cumulative[frame_index] = pairwise_tmats[frame_index] @ cumulative[frame_index - 1]
    return cumulative
