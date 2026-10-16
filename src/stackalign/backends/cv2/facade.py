"""Compatibility facade over the shared RegisterModel implementation."""

from __future__ import annotations

from numpy.typing import NDArray

from stackalign.backends.execution import CancelCallback, ExecutionMode, ProgressCallback
from stackalign.backends.models import TransformModel
from stackalign.constants import Method, ReferenceStrategy


class Cv2Backend:
    def __init__(self, *, execution: ExecutionMode | None = None,
                 workers: int | None = None) -> None:
        """Set the backend and explicit execution options for registration."""
        self.execution: ExecutionMode | None = execution
        self.workers = workers

    def fit_time(self, array: NDArray, axes: str, method: Method = "translation",
                 reference_strategy: ReferenceStrategy = "first",
                 fit_channel: int | None = None, *,
                 progress: ProgressCallback | None = None,
                 cancelled: CancelCallback | None = None) -> TransformModel:
        """Fit one global transform per frame using the selected reference strategy."""
        from stackalign.api import RegisterModel
        return RegisterModel("cv2", execution=self.execution, workers=self.workers).fit_time(
            array, axes, method, reference_strategy, fit_channel,
            progress=progress, cancelled=cancelled).model

    def fit_channel(self, array: NDArray, axes: str, method: Method = "translation",
                    reference_channel: int | None = None, reference_frame: int = 0, *,
                    progress: ProgressCallback | None = None,
                    cancelled: CancelCallback | None = None) -> TransformModel:
        """Fit channel transforms at the selected reference frame."""
        from stackalign.api import RegisterModel
        return RegisterModel("cv2", execution=self.execution, workers=self.workers).fit_channel(
            array, axes, method, reference_channel, reference_frame,
            progress=progress, cancelled=cancelled).model

    def apply(self, array: NDArray, axes: str, model: TransformModel, *,
              out: NDArray | None = None, progress: ProgressCallback | None = None,
              cancelled: CancelCallback | None = None) -> NDArray:
        """Apply a fitted model while preserving the input axes and dtype."""
        from stackalign.api import RegisterModel
        return RegisterModel("cv2", execution=self.execution, workers=self.workers).set_model(
            model).apply(array, axes, out=out, progress=progress, cancelled=cancelled)

    def apply_channel(self, array: NDArray, axes: str, model: TransformModel) -> NDArray:
        """Apply channel transforms and leave the reference channel unchanged."""
        if model.mode != "channel":
            raise ValueError("apply_channel requires a channel model.")
        return self.apply(array, axes, model)
