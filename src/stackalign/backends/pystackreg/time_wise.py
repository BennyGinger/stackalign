from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from stackalign.backends.models import TransformModel
from stackalign.constants import Method, ReferenceStrategy
from stackalign.preparation import ApplyPreparation, FitPreparation
from stackalign.backends.pystackreg.execution import apply_tyx_substack, fit_frames_to_reference, fit_previous_pairwise_tmats
from stackalign.backends.pystackreg.utils import accumulate_pairwise_tmats, validate_method, validate_reference_strategy


def fit_time(array: NDArray[np.generic], axes: str, method: Method = "translation", reference_strategy: ReferenceStrategy = "first", fit_channel: int | None = None) -> TransformModel:
    """
    Fit time-wise transforms as explicit per-frame transformation matrices.
    """
    validate_method(method)
    validate_reference_strategy(reference_strategy)

    preparation = FitPreparation.for_time(array=array, axes=axes, fit_channel=fit_channel)
    fit_array = preparation.fit_array
    if fit_array is None:
        raise RuntimeError("Internal error: fit array was not prepared for fit_time().")

    tmats = _fit_time_tmats(fit_array, method=method, reference_strategy=reference_strategy)
    return TransformModel(mode="time", method=method, transform=tmats)


def apply_time(array: NDArray[np.generic], axes: str, model: TransformModel) -> NDArray[np.generic]:
    """
    Apply a fitted time-wise model to all TYX substacks in the input array.
    """
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
        transformed_apply[slicer] = apply_tyx_substack(substack_tyx, tmats, model.method)

    return preparation.restore_apply_output(transformed_apply)


def _fit_time_tmats(fit_array_tyx: NDArray[np.float32], method: Method, reference_strategy: ReferenceStrategy) -> NDArray[np.float64]:
    if fit_array_tyx.shape[0] == 0:
        raise ValueError("fit_time requires at least one frame. Got T=0.")

    if reference_strategy == "first":
        reference = np.asarray(fit_array_tyx[0], dtype=np.float32)
        return fit_frames_to_reference(fit_array_tyx, reference=reference, method=method)

    if reference_strategy == "mean":
        reference = np.asarray(fit_array_tyx.mean(axis=0), dtype=np.float32)
        return fit_frames_to_reference(fit_array_tyx, reference=reference, method=method)

    if reference_strategy == "previous":
        pairwise_tmats = fit_previous_pairwise_tmats(fit_array_tyx, method=method)
        return accumulate_pairwise_tmats(pairwise_tmats)

    raise ValueError(f"Unsupported reference_strategy='{reference_strategy}'.")


