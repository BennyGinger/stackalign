from __future__ import annotations

from functools import partial

import cv2
import numpy as np
from numpy.typing import NDArray

from stackalign.backends.execution import apply_cyx_substack
from stackalign.backends.models import TransformModel
from stackalign.backends.transforms import identity_tmats
from stackalign.constants import Method
from stackalign.preparation import ApplyPreparation, FitPreparation

from .time_wise import _ecc_fit_to_tmat
from .utils import tmat_to_cv2_warp, validate_method


def fit_channel(array: NDArray[np.generic], axes: str, method: Method = "translation", reference_channel: int | None = None, reference_frame: int = 0) -> TransformModel:
    validate_method(method)

    preparation = FitPreparation.for_channel(
        array=array,
        axes=axes,
        reference_channel=reference_channel,
        reference_frame=reference_frame,
    )
    fit_array = preparation.fit_array
    if fit_array is None:
        raise RuntimeError("Internal error: fit array was not prepared for fit_channel().")
    if reference_channel is None:
        raise ValueError("fit_channel requires reference_channel.")

    c_len = fit_array.shape[0]
    tmats = identity_tmats(c_len)
    reference_image = np.asarray(fit_array[reference_channel], dtype=np.float32)

    for channel_index in range(c_len):
        if channel_index == reference_channel:
            continue

        moving_image = np.asarray(fit_array[channel_index], dtype=np.float32)
        tmats[channel_index] = _ecc_fit_to_tmat(
            reference=reference_image,
            moving=moving_image,
            method=method,
            frame_index=channel_index,
        )

    return TransformModel(
        mode="channel",
        method=method,
        transform=tmats,
        reference_channel=reference_channel,
    )


def apply_channel(array: NDArray[np.generic], axes: str, model: TransformModel) -> NDArray[np.generic]:
    if model.mode != "channel":
        raise ValueError(f"apply_channel requires a channel model. Got mode='{model.mode}'.")
    if not isinstance(model.transform, np.ndarray):
        raise TypeError("apply_channel expects model.transform to be a matrix stack with shape (C, 3, 3).")
    if model.reference_channel is None:
        raise ValueError("apply_channel requires a model with reference_channel set.")

    tmats = np.asarray(model.transform, dtype=np.float64)
    if tmats.ndim != 3 or tmats.shape[1:] != (3, 3):
        raise ValueError(f"channel model transform must have shape (C, 3, 3). Got {tmats.shape}.")

    preparation = ApplyPreparation.for_channel(array=array, axes=axes)
    transformed_apply = np.empty(preparation.apply_array.shape, dtype=np.float32)
    c_len = preparation.apply_array.shape[preparation.apply_axes.index("C")]

    if tmats.shape[0] != c_len:
        raise ValueError(
            f"Channel model length must match array C length. Got {tmats.shape[0]} transforms for C={c_len}."
        )

    for slicer, substack_cyx in preparation.iter_apply_cyx_substacks():
        transformed_apply[slicer] = apply_cyx_substack(
            substack_cyx,
            tmats,
            model.reference_channel,
            partial(_apply_channel_image_task, method=model.method),
        )

    return preparation.restore_apply_output(transformed_apply)


def _apply_channel_image_task(channel_index: int, image: NDArray[np.float32], tmat: NDArray[np.float64], method: Method) -> tuple[int, NDArray[np.float32]]:
    warp = tmat_to_cv2_warp(tmat, method=method)
    image_float = np.asarray(image, dtype=np.float32)
    height, width = image_float.shape

    transformed = cv2.warpAffine(
        image_float,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0,),
    )
    return channel_index, np.asarray(transformed, dtype=np.float32)
