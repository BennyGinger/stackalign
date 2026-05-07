from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from stackalign.backends.cv2 import Cv2Backend


def _center_of_mass(image: np.ndarray) -> tuple[float, float]:
    weights = image.astype(np.float64)
    total = float(weights.sum())
    if total == 0.0:
        raise ValueError("center_of_mass requires non-empty signal.")
    yy, xx = np.indices(weights.shape)
    return float((yy * weights).sum() / total), float((xx * weights).sum() / total)


def _gaussian_blob(shape: tuple[int, int], center_yx: tuple[float, float], sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    cy, cx = center_yx
    blob = np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2.0 * sigma * sigma))
    return (blob * 1000.0).astype(np.float32)


def _cv2_cyx_stack() -> np.ndarray:
    reference = _gaussian_blob((64, 64), center_yx=(28.0, 28.0), sigma=3.0)
    reference += _gaussian_blob((64, 64), center_yx=(18.0, 42.0), sigma=4.0)
    shifted_a = np.roll(reference, shift=(2, -3), axis=(0, 1))
    shifted_b = np.roll(reference, shift=(-3, 4), axis=(0, 1))
    return np.stack([reference, shifted_a, shifted_b], axis=0).astype(np.uint16)


def test_cv2_fit_channel_translation_works_on_cyx() -> None:
    backend = Cv2Backend()
    stack = _cv2_cyx_stack()
    model = backend.fit_channel(
        stack,
        axes="CYX",
        method="translation",
        reference_channel=0,
    )

    assert model.mode == "channel"
    assert model.method == "translation"
    assert model.reference_channel == 0
    assert isinstance(model.transform, np.ndarray)
    assert model.transform.shape == (stack.shape[0], 3, 3)
    npt.assert_allclose(model.transform[0], np.eye(3, dtype=np.float64), atol=1e-8)


def test_cv2_apply_channel_preserves_shape_dtype() -> None:
    backend = Cv2Backend()
    stack = _cv2_cyx_stack()
    model = backend.fit_channel(
        stack,
        axes="CYX",
        method="translation",
        reference_channel=0,
    )
    result = backend.apply(stack, axes="CYX", model=model)

    assert result.shape == stack.shape
    assert result.dtype == stack.dtype


def test_cv2_shifted_channel_aligns_to_reference() -> None:
    backend = Cv2Backend()
    stack = _cv2_cyx_stack()
    model = backend.fit_channel(
        stack,
        axes="CYX",
        method="translation",
        reference_channel=0,
    )
    result = backend.apply(stack, axes="CYX", model=model)

    ref_center = _center_of_mass(result[0])
    shifted_center = _center_of_mass(result[1])
    assert abs(shifted_center[0] - ref_center[0]) < 0.5
    assert abs(shifted_center[1] - ref_center[1]) < 0.5


def test_cv2_fit_channel_rigid_body_smoke() -> None:
    backend = Cv2Backend()
    stack = _cv2_cyx_stack()
    model = backend.fit_channel(
        stack,
        axes="CYX",
        method="rigid_body",
        reference_channel=0,
    )
    result = backend.apply(stack, axes="CYX", model=model)

    assert model.transform.shape == (stack.shape[0], 3, 3)
    assert result.shape == stack.shape


def test_cv2_fit_channel_affine_smoke() -> None:
    backend = Cv2Backend()
    stack = _cv2_cyx_stack()
    model = backend.fit_channel(
        stack,
        axes="CYX",
        method="affine",
        reference_channel=0,
    )
    result = backend.apply(stack, axes="CYX", model=model)

    assert model.transform.shape == (stack.shape[0], 3, 3)
    assert result.shape == stack.shape


def test_cv2_fit_channel_invalid_method_raises() -> None:
    backend = Cv2Backend()
    stack = _cv2_cyx_stack()
    with pytest.raises(ValueError, match="Unsupported method"):
        backend.fit_channel(
            stack,
            axes="CYX",
            method="projective",  # type: ignore[arg-type]
            reference_channel=0,
        )


def test_cv2_fit_channel_missing_reference_channel_raises() -> None:
    backend = Cv2Backend()
    stack = _cv2_cyx_stack()
    with pytest.raises(ValueError, match="reference_channel"):
        backend.fit_channel(
            stack,
            axes="CYX",
            method="translation",
            reference_channel=None,
        )
