from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from stackalign.backends import Cv2Backend, get_backend
from stackalign.backends.cv2.utils import cv2_warp_to_tmat, tmat_to_cv2_warp


def _gaussian_blob(shape: tuple[int, int], center_yx: tuple[float, float], sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    cy, cx = center_yx
    blob = np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2.0 * sigma * sigma))
    return (blob * 1000.0).astype(np.float32)


def _synthetic_tyx_stack() -> np.ndarray:
    base = _gaussian_blob((64, 64), center_yx=(30.0, 28.0), sigma=3.0)
    # Add a second feature to break symmetry and help ECC convergence.
    base += _gaussian_blob((64, 64), center_yx=(18.0, 42.0), sigma=4.5)
    frames = [
        np.roll(base, shift=(0, 0), axis=(0, 1)),
        np.roll(base, shift=(1, -1), axis=(0, 1)),
        np.roll(base, shift=(2, -2), axis=(0, 1)),
        np.roll(base, shift=(3, -3), axis=(0, 1)),
    ]
    return np.stack(frames, axis=0).astype(np.uint16)


def test_get_backend_returns_cv2_backend() -> None:
    backend = get_backend("cv2")
    assert isinstance(backend, Cv2Backend)


def test_cv2_warp_matrix_conversion_roundtrip() -> None:
    warp = np.array([[1.0, 0.0, -2.5], [0.0, 1.0, 4.0]], dtype=np.float32)
    tmat = cv2_warp_to_tmat(warp)
    warp_roundtrip = tmat_to_cv2_warp(tmat, method="translation")

    npt.assert_allclose(warp_roundtrip, warp, rtol=0.0, atol=1e-6)


def test_cv2_fit_time_translation_first_works() -> None:
    backend = Cv2Backend()
    stack = _synthetic_tyx_stack()
    model = backend.fit_time(
        stack,
        axes="TYX",
        method="translation",
        reference_strategy="first",
    )

    assert model.mode == "time"
    assert model.method == "translation"
    assert model.transform.shape == (stack.shape[0], 3, 3)


def test_cv2_apply_preserves_shape_and_dtype() -> None:
    backend = Cv2Backend()
    stack = _synthetic_tyx_stack()
    model = backend.fit_time(stack, axes="TYX", method="translation", reference_strategy="first")
    result = backend.apply(stack, axes="TYX", model=model)

    assert result.shape == stack.shape
    assert result.dtype == stack.dtype


def test_cv2_invalid_method_raises() -> None:
    backend = Cv2Backend()
    stack = _synthetic_tyx_stack()
    with pytest.raises(ValueError, match="Unsupported method"):
        backend.fit_time(stack, axes="TYX", method="projective")  # type: ignore[arg-type]


def test_cv2_previous_accumulates_pairwise_into_frame0() -> None:
    backend = Cv2Backend()
    base = _gaussian_blob((48, 48), center_yx=(20.0, 20.0))
    frames = [
        np.roll(base, shift=(0, 0), axis=(0, 1)),
        np.roll(base, shift=(2, -1), axis=(0, 1)),
        np.roll(base, shift=(4, -2), axis=(0, 1)),
    ]
    stack = np.stack(frames, axis=0).astype(np.uint16)

    model = backend.fit_time(stack, axes="TYX", method="translation", reference_strategy="previous")

    shift_frame1_y = model.transform[1, 1, 2]
    shift_frame2_y = model.transform[2, 1, 2]
    assert abs(shift_frame2_y) > abs(shift_frame1_y) * 1.5


def test_cv2_translation_blob_aligns_after_apply() -> None:
    backend = Cv2Backend()
    base = _gaussian_blob((64, 64), center_yx=(30.0, 28.0))
    shifted = np.roll(base, shift=(5, -3), axis=(0, 1))
    stack = np.stack([base, shifted], axis=0).astype(np.uint16)

    model = backend.fit_time(stack, axes="TYX", method="translation", reference_strategy="first")
    aligned = backend.apply(stack, axes="TYX", model=model).astype(np.float32)

    ref_peak = np.unravel_index(np.argmax(aligned[0]), aligned[0].shape)
    aligned_peak = np.unravel_index(np.argmax(aligned[1]), aligned[1].shape)
    assert abs(ref_peak[0] - aligned_peak[0]) <= 1
    assert abs(ref_peak[1] - aligned_peak[1]) <= 1


def test_cv2_rigid_body_smoke() -> None:
    backend = Cv2Backend()
    stack = _synthetic_tyx_stack()
    model = backend.fit_time(stack, axes="TYX", method="rigid_body", reference_strategy="first")
    result = backend.apply(stack, axes="TYX", model=model)

    assert model.transform.shape == (stack.shape[0], 3, 3)
    assert result.shape == stack.shape


def test_cv2_affine_smoke() -> None:
    backend = Cv2Backend()
    stack = _synthetic_tyx_stack()
    model = backend.fit_time(stack, axes="TYX", method="affine", reference_strategy="first")
    result = backend.apply(stack, axes="TYX", model=model)

    assert model.transform.shape == (stack.shape[0], 3, 3)
    assert result.shape == stack.shape


# Channel-wise behavior is covered in tests/backends/cv2/test_channel_wise_cv2.py.
