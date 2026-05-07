from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pystackreg import StackReg

from stackalign.constants import Method

_METHOD_TO_STACKREG = {
    "translation": StackReg.TRANSLATION,
    "rigid_body": StackReg.RIGID_BODY,
    "affine": StackReg.AFFINE,
}


def validate_method(method: str) -> None:
    if method not in _METHOD_TO_STACKREG:
        raise ValueError(f"Unsupported method='{method}'. Supported methods: {sorted(_METHOD_TO_STACKREG)}")


def create_stackreg(method: Method) -> StackReg:
    return StackReg(_METHOD_TO_STACKREG[method])


def register_frame_to_reference_task(frame_index: int, reference: NDArray[np.float32], moving: NDArray[np.float32], method: Method) -> tuple[int, NDArray[np.float64]]:
    sr = create_stackreg(method)
    return frame_index, sr.register(reference, moving)


def register_previous_pair_task(frame_index: int, previous: NDArray[np.float32], current: NDArray[np.float32], method: Method) -> tuple[int, NDArray[np.float64]]:
    sr = create_stackreg(method)
    return frame_index, sr.register(previous, current)


def apply_frame_tmat_task(frame_index: int, frame: NDArray[np.float32], tmat: NDArray[np.float64], method: Method) -> tuple[int, NDArray[np.float32]]:
    sr = create_stackreg(method)
    return frame_index, np.asarray(sr.transform(frame, tmat=tmat), dtype=np.float32)


def apply_channel_image_task(channel_index: int, image: NDArray[np.float32], tmat: NDArray[np.float64], method: Method) -> tuple[int, NDArray[np.float32]]:
    sr = create_stackreg(method)
    return channel_index, np.asarray(sr.transform(image, tmat=tmat), dtype=np.float32)
