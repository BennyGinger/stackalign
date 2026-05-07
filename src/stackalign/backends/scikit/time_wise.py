from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as ndimage_shift
from skimage.registration import phase_cross_correlation

from stackalign.backends.execution import apply_tyx_substack, fit_frames_to_reference, fit_previous_pairwise
from stackalign.backends.models import TransformModel
from stackalign.backends.transforms import accumulate_pairwise_tmats, validate_reference_strategy
from stackalign.constants import ReferenceStrategy
from stackalign.preparation import ApplyPreparation, FitPreparation

from .utils import shift_to_tmat, validate_method


def fit_time(array: NDArray[np.generic], axes: str, method: str = "translation", reference_strategy: ReferenceStrategy = "first", fit_channel: int | None = None) -> TransformModel:
    validate_method(method)
    validate_reference_strategy(reference_strategy)

    preparation = FitPreparation.for_time(array=array, axes=axes, fit_channel=fit_channel)
    fit_array = preparation.fit_array
    if fit_array is None:
        raise RuntimeError("Internal error: fit array was not prepared for fit_time().")

    tmats = _fit_time_tmats(fit_array, reference_strategy=reference_strategy)
    return TransformModel(mode="time", method=method, transform=tmats)  # type: ignore[arg-type]


def apply_time(array: NDArray[np.generic], axes: str, model: TransformModel) -> NDArray[np.generic]:
    if model.mode != "time":
        raise ValueError(f"apply_time requires a time model. Got mode='{model.mode}'.")
    if not isinstance(model.transform, np.ndarray):
        raise TypeError("apply_time expects model.transform to be a matrix stack with shape (T, 3, 3).")

    tmats = np.asarray(model.transform, dtype=np.float64)
    if tmats.ndim != 3 or tmats.shape[1:] != (3, 3):
        raise ValueError(f"time model transform must have shape (T, 3, 3). Got {tmats.shape}.")

    preparation = ApplyPreparation.for_time(array=array, axes=axes)
    transformed_apply = np.empty(preparation.apply_array.shape, dtype=np.float32)

    for slicer, substack_tyx in preparation.iter_apply_tyx_substacks():
        transformed_apply[slicer] = apply_tyx_substack(substack_tyx, tmats, _apply_frame_task)

    return preparation.restore_apply_output(transformed_apply)

# ---------------------------------------------------------------------------
# Process-safe top-level worker functions
# ---------------------------------------------------------------------------

def _fit_frame_to_reference_task(frame_index: int, reference: NDArray[np.float32], moving: NDArray[np.float32]) -> tuple[int, NDArray[np.float64]]:
    shift, *_ = phase_cross_correlation(reference, moving, normalization=None) # type: ignore[no-untyped-call]
    return frame_index, shift_to_tmat(shift)


def _fit_previous_pair_task(frame_index: int, previous: NDArray[np.float32], current: NDArray[np.float32]) -> tuple[int, NDArray[np.float64]]:
    shift, *_ = phase_cross_correlation(previous, current, normalization=None) # type: ignore[no-untyped-call]
    return frame_index, shift_to_tmat(shift)


def _apply_frame_task(frame_index: int, frame: NDArray[np.float32], tmat: NDArray[np.float64]) -> tuple[int, NDArray[np.float32]]:
    shift_yx = (tmat[1, 2], tmat[0, 2])
    return frame_index, ndimage_shift(frame, shift=shift_yx).astype(np.float32)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _fit_time_tmats(fit_array_tyx: NDArray[np.float32], reference_strategy: ReferenceStrategy) -> NDArray[np.float64]:
    if fit_array_tyx.shape[0] == 0:
        raise ValueError("fit_time requires at least one frame. Got T=0.")

    if reference_strategy == "first":
        reference = np.asarray(fit_array_tyx[0], dtype=np.float32)
        return fit_frames_to_reference(fit_array_tyx, reference, _fit_frame_to_reference_task)

    if reference_strategy == "mean":
        reference = np.asarray(fit_array_tyx.mean(axis=0), dtype=np.float32)
        return fit_frames_to_reference(fit_array_tyx, reference, _fit_frame_to_reference_task)

    if reference_strategy == "previous":
        pairwise = fit_previous_pairwise(fit_array_tyx, _fit_previous_pair_task)
        return accumulate_pairwise_tmats(pairwise)

    raise ValueError(f"Unsupported reference_strategy='{reference_strategy}'.")


