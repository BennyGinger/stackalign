from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from stackalign.constants import Method, ReferenceStrategy
from stackalign.backends.models import TransformModel


class FrameApplyFn(Protocol):
    def __call__(self, frame_index: int, frame: NDArray[np.float32], tmat: NDArray[np.float64]) -> tuple[int, NDArray[np.float32]]:
        """Transform one frame and return its index and pixels."""
        ...


class ChannelApplyFn(Protocol):
    def __call__(self, channel_index: int, image: NDArray[np.float32], tmat: NDArray[np.float64]) -> tuple[int, NDArray[np.float32]]:
        """Transform one channel image and return its index and pixels."""
        ...


class FrameFitFn(Protocol):
    def __call__(self, frame_index: int, reference: NDArray[np.float32], moving: NDArray[np.float32]) -> tuple[int, NDArray[np.float64]]:
        """Fit a frame to a reference and return its index and matrix."""
        ...


class PairFitFn(Protocol):
    def __call__(self, frame_index: int, previous: NDArray[np.float32], current: NDArray[np.float32]) -> tuple[int, NDArray[np.float64]]:
        """Fit an adjacent-frame pair and return its index and matrix."""
        ...


@runtime_checkable
class Backend(Protocol):
    """
    Interface contract for registration backends.
    """
    def fit_time(self, array: NDArray[np.generic], axes: str, method: Method = "translation", reference_strategy: ReferenceStrategy = "first", fit_channel: int | None = None) -> TransformModel:
        """Fit one global transform per frame using the selected reference strategy."""
        ...

    def apply(self, array: NDArray[np.generic], axes: str, model: TransformModel) -> NDArray[np.generic]:
        """Apply a fitted model while preserving the input axes and dtype."""
        ...

    def fit_channel(self, array: NDArray[np.generic], axes: str, method: Method = "translation", reference_channel: int | None = None, reference_frame: int = 0) -> TransformModel:
        """Fit channel transforms at the selected reference frame."""
        ...
