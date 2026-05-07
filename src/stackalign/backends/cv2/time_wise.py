from __future__ import annotations

from functools import partial

import cv2
import numpy as np
from numpy.typing import NDArray

from stackalign.backends.execution import apply_tyx_substack, fit_frames_to_reference, fit_previous_pairwise
from stackalign.backends.models import TransformModel
from stackalign.backends.transforms import accumulate_pairwise_tmats, validate_reference_strategy
from stackalign.constants import Method, ReferenceStrategy
from stackalign.preparation import ApplyPreparation, FitPreparation

from .utils import build_ecc_criteria, create_initial_warp, cv2_warp_to_tmat, get_motion_model, normalize_for_ecc, tmat_to_cv2_warp, validate_method


def fit_time(array: NDArray[np.generic], axes: str, method: Method = "translation", reference_strategy: ReferenceStrategy = "first", fit_channel: int | None = None) -> TransformModel:
    validate_method(method)
    validate_reference_strategy(reference_strategy)

    preparation = FitPreparation.for_time(array=array, axes=axes, fit_channel=fit_channel)
    fit_array = preparation.fit_array
    if fit_array is None:
        raise RuntimeError("Internal error: fit array was not prepared for fit_time().")

    tmats = _fit_time_tmats(fit_array, method=method, reference_strategy=reference_strategy)
    return TransformModel(mode="time", method=method, transform=tmats)


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
        transformed_apply[slicer] = apply_tyx_substack(
            substack_tyx,
            tmats,
            partial(_apply_frame_task, method=model.method),
        )

    return preparation.restore_apply_output(transformed_apply)


# ---------------------------------------------------------------------------
# Process-safe top-level worker functions
# ---------------------------------------------------------------------------

def _fit_frame_to_reference_task(frame_index: int, reference: NDArray[np.float32], moving: NDArray[np.float32], method: Method) -> tuple[int, NDArray[np.float64]]:
    return frame_index, _ecc_fit_to_tmat(reference, moving, method=method, frame_index=frame_index)


def _fit_previous_pair_task(frame_index: int, previous: NDArray[np.float32], current: NDArray[np.float32], method: Method) -> tuple[int, NDArray[np.float64]]:
    return frame_index, _ecc_fit_to_tmat(previous, current, method=method, frame_index=frame_index)


def _apply_frame_task(frame_index: int, frame: NDArray[np.float32], tmat: NDArray[np.float64], method: Method) -> tuple[int, NDArray[np.float32]]:
    warp = tmat_to_cv2_warp(tmat, method=method)
    frame_float = np.asarray(frame, dtype=np.float32)
    height, width = frame_float.shape

    transformed = cv2.warpAffine(
        frame_float,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0,),
    )
    return frame_index, np.asarray(transformed, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _fit_time_tmats(fit_array_tyx: NDArray[np.float32], method: Method, reference_strategy: ReferenceStrategy) -> NDArray[np.float64]:
    if fit_array_tyx.shape[0] == 0:
        raise ValueError("fit_time requires at least one frame. Got T=0.")

    if reference_strategy == "first":
        reference = np.asarray(fit_array_tyx[0], dtype=np.float32)
        return fit_frames_to_reference(
            fit_array_tyx,
            reference,
            partial(_fit_frame_to_reference_task, method=method),
        )

    if reference_strategy == "mean":
        reference = np.asarray(fit_array_tyx.mean(axis=0), dtype=np.float32)
        return fit_frames_to_reference(
            fit_array_tyx,
            reference,
            partial(_fit_frame_to_reference_task, method=method),
        )

    if reference_strategy == "previous":
        pairwise_tmats = fit_previous_pairwise(
            fit_array_tyx,
            partial(_fit_previous_pair_task, method=method),
        )
        return accumulate_pairwise_tmats(pairwise_tmats)

    raise ValueError(f"Unsupported reference_strategy='{reference_strategy}'.")


def _ecc_fit_to_tmat(reference: NDArray[np.float32], moving: NDArray[np.float32], method: Method, frame_index: int) -> NDArray[np.float64]:
    reference_norm = normalize_for_ecc(reference)
    moving_norm = normalize_for_ecc(moving)
    warp = create_initial_warp(method)

    try:
        _, estimated_warp = cv2.findTransformECC(
            reference_norm,
            moving_norm,
            warp,
            get_motion_model(method),
            build_ecc_criteria(),
            np.empty((0, 0), dtype=np.uint8),
            1,
        )
    except cv2.error as exc:  # pragma: no cover - depends on OpenCV internals
        raise RuntimeError(f"ECC fit failed for frame {frame_index} using method='{method}': {exc}") from exc

    return cv2_warp_to_tmat(np.asarray(estimated_warp, dtype=np.float32))
