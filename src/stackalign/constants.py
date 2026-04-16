from __future__ import annotations

from typing import Final, Literal, TypeVar

from numpy.typing import NDArray
import numpy as np

DType = TypeVar("DType", bound=np.generic)

AXIS_LABELS: Final[frozenset[str]] = frozenset({"T", "C", "Z", "Y", "X"})
CANONICAL_AXIS_ORDER: Final[str] = "TCZYX"

Method = Literal["translation", "rigid_body", "affine"]
ReferenceStrategy = Literal["first", "previous", "mean"]