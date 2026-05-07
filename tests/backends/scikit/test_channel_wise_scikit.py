from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from stackalign.backends.scikit import ScikitBackend


def _center_of_mass(image: np.ndarray) -> tuple[float, float]:
    weights = image.astype(np.float64)
    total = float(weights.sum())
    if total == 0.0:
        raise ValueError("center_of_mass requires non-empty signal.")
    yy, xx = np.indices(weights.shape)
    return float((yy * weights).sum() / total), float((xx * weights).sum() / total)


def test_scikit_fit_channel_translation_works_on_cyx(translated_cyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    model = backend.fit_channel(
        translated_cyx_stack,
        axes="CYX",
        method="translation",
        reference_channel=0,
    )

    assert model.mode == "channel"
    assert model.method == "translation"
    assert model.reference_channel == 0
    assert isinstance(model.transform, np.ndarray)
    assert model.transform.shape == (translated_cyx_stack.shape[0], 3, 3)
    npt.assert_allclose(model.transform[0], np.eye(3, dtype=np.float64), atol=1e-8)


def test_scikit_apply_channel_preserves_shape_dtype(translated_tcyx_channel_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    model = backend.fit_channel(
        translated_tcyx_channel_stack,
        axes="TCYX",
        method="translation",
        reference_channel=0,
        reference_frame=0,
    )
    result = backend.apply(translated_tcyx_channel_stack, axes="TCYX", model=model)

    assert result.shape == translated_tcyx_channel_stack.shape
    assert result.dtype == translated_tcyx_channel_stack.dtype


def test_scikit_shifted_channel_aligns_to_reference(translated_cyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    model = backend.fit_channel(
        translated_cyx_stack,
        axes="CYX",
        method="translation",
        reference_channel=0,
    )
    result = backend.apply(translated_cyx_stack, axes="CYX", model=model)

    ref_center = _center_of_mass(result[0])
    shifted_center = _center_of_mass(result[1])
    assert abs(shifted_center[0] - ref_center[0]) < 0.25
    assert abs(shifted_center[1] - ref_center[1]) < 0.25


def test_scikit_fit_channel_invalid_method_raises(translated_cyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    with pytest.raises(ValueError, match="only supports method='translation'"):
        backend.fit_channel(
            translated_cyx_stack,
            axes="CYX",
            method="affine",  # type: ignore[arg-type]
            reference_channel=0,
        )


def test_scikit_fit_channel_missing_reference_channel_raises(translated_cyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    with pytest.raises(ValueError, match="reference_channel"):
        backend.fit_channel(
            translated_cyx_stack,
            axes="CYX",
            method="translation",
            reference_channel=None,
        )
