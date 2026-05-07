from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from stackalign.backends.models import TransformModel
from stackalign.constants import Method, ReferenceStrategy

from .channel_wise import apply_channel, fit_channel
from .time_wise import apply_time, fit_time


class Cv2Backend:
    """Thin backend facade for OpenCV ECC registration."""

    def fit_time(self, array: NDArray[np.generic], axes: str, method: Method = "translation", reference_strategy: ReferenceStrategy = "first", fit_channel: int | None = None,) -> TransformModel:
        return fit_time(
            array=array,
            axes=axes,
            method=method,
            reference_strategy=reference_strategy,
            fit_channel=fit_channel,
        )

    def apply(self, array: NDArray[np.generic], axes: str, model: TransformModel) -> NDArray[np.generic]:
        if model.mode == "time":
            return apply_time(array=array, axes=axes, model=model)

        if model.mode == "channel":
            return apply_channel(array=array, axes=axes, model=model)

        raise RuntimeError(f"Unsupported transform mode '{model.mode}'.")

    def fit_channel(self, array: NDArray[np.generic], axes: str, method: Method = "translation", reference_channel: int | None = None, reference_frame: int = 0) -> TransformModel:
        return fit_channel(
            array=array,
            axes=axes,
            method=method,
            reference_channel=reference_channel,
            reference_frame=reference_frame,
        )

    def apply_channel(self, array: NDArray[np.generic], axes: str, model: TransformModel) -> NDArray[np.generic]:
        return apply_channel(array=array, axes=axes, model=model)
