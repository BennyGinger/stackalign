from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from stackalign.constants import Method, ReferenceStrategy
from stackalign.backends.models import TransformModel


class FrameApplyFn(Protocol):
    def __call__(self, frame_index: int, frame: NDArray[np.float32], tmat: NDArray[np.float64]) -> tuple[int, NDArray[np.float32]]: ...


class ChannelApplyFn(Protocol):
    def __call__(self, channel_index: int, image: NDArray[np.float32], tmat: NDArray[np.float64]) -> tuple[int, NDArray[np.float32]]: ...


class FrameFitFn(Protocol):
    def __call__(self, frame_index: int, reference: NDArray[np.float32], moving: NDArray[np.float32]) -> tuple[int, NDArray[np.float64]]: ...


class PairFitFn(Protocol):
    def __call__(self, frame_index: int, previous: NDArray[np.float32], current: NDArray[np.float32]) -> tuple[int, NDArray[np.float64]]: ...


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
