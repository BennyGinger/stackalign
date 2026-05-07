from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from stackalign.constants import Method

ECC_MAX_ITER = 100
ECC_EPSILON = 1e-5

_METHOD_TO_CV2_MOTION: dict[str, int] = {
    "translation": cv2.MOTION_TRANSLATION,
    "rigid_body": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
}


def validate_method(method: str) -> None:
    if method not in _METHOD_TO_CV2_MOTION:
        raise ValueError(f"Unsupported method='{method}'. Supported methods: {sorted(_METHOD_TO_CV2_MOTION)}")


def get_motion_model(method: Method) -> int:
    return _METHOD_TO_CV2_MOTION[method]


def create_initial_warp(method: Method) -> NDArray[np.float32]:
    validate_method(method)
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def cv2_warp_to_tmat(warp_matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    warp = np.asarray(warp_matrix, dtype=np.float64)
    if warp.shape != (2, 3):
        raise ValueError(f"warp_matrix must have shape (2, 3). Got {warp.shape}.")

    tmat = np.eye(3, dtype=np.float64)
    tmat[0, 0] = warp[0, 0]
    tmat[0, 1] = warp[0, 1]
    tmat[0, 2] = warp[0, 2]
    tmat[1, 0] = warp[1, 0]
    tmat[1, 1] = warp[1, 1]
    tmat[1, 2] = warp[1, 2]
    return tmat


def tmat_to_cv2_warp(tmat: NDArray[np.floating], method: Method) -> NDArray[np.float32]:
    validate_method(method)

    tmat_3x3 = np.asarray(tmat, dtype=np.float64)
    if tmat_3x3.shape != (3, 3):
        raise ValueError(f"tmat must have shape (3, 3). Got {tmat_3x3.shape}.")

    return tmat_3x3[:2, :].astype(np.float32)


def normalize_for_ecc(image: NDArray[np.floating] | NDArray[np.integer]) -> NDArray[np.float32]:
    normalized = np.asarray(image, dtype=np.float32)
    normalized = normalized - normalized.min()
    max_value = normalized.max()
    if max_value > 0:
        normalized = normalized / max_value
    return normalized


def build_ecc_criteria() -> tuple[int, int, float]:
    return (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_MAX_ITER,
        ECC_EPSILON,
    )
