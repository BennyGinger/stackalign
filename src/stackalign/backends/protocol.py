from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from stackalign.constants import Method, ReferenceStrategy
from stackalign.backends.models import TransformModel


@runtime_checkable
class Backend(Protocol):
    """
    Interface contract for registration backends.
    """
    def fit_time(self, array: NDArray[np.generic], axes: str, method: Method = "translation", reference_strategy: ReferenceStrategy = "first", fit_channel: int | None = None) -> TransformModel:
        ...

    def apply(self, array: NDArray[np.generic], axes: str, model: TransformModel) -> NDArray[np.generic]:
        ...

    def fit_channel(self, array: NDArray[np.generic], axes: str, method: Method = "translation", reference_channel: int | None = None, reference_frame: int = 0) -> TransformModel:
        ...
