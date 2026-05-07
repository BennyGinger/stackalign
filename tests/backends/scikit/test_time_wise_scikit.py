from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from stackalign.backends import ScikitBackend, get_backend
from stackalign.backends.scikit.utils import shift_to_tmat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _blob_frame(shape: tuple[int, int], top: int, left: int, size: int = 5, value: float = 1000.0) -> np.ndarray:
    frame = np.zeros(shape, dtype=np.float32)
    frame[top : top + size, left : left + size] = value
    return frame


@pytest.fixture
def synthetic_tyx_shift3() -> np.ndarray:
    """4-frame TYX stack where each frame shifts +3 in y relative to frame 0."""
    frames = [
        _blob_frame((40, 40), top=5, left=5),
        _blob_frame((40, 40), top=8, left=5),
        _blob_frame((40, 40), top=11, left=5),
        _blob_frame((40, 40), top=14, left=5),
    ]
    return np.stack(frames, axis=0).astype(np.uint16)


@pytest.fixture
def synthetic_tyx_pairwise() -> np.ndarray:
    """3-frame TYX stack with uniform +2 y-shift per frame."""
    frames = [
        _blob_frame((40, 40), top=5, left=5),
        _blob_frame((40, 40), top=7, left=5),
        _blob_frame((40, 40), top=9, left=5),
    ]
    return np.stack(frames, axis=0).astype(np.uint16)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def test_get_backend_returns_scikit_backend() -> None:
    backend = get_backend("scikit")
    assert isinstance(backend, ScikitBackend)


def test_get_backend_pystackreg_still_works() -> None:
    from stackalign.backends import PystackregBackend
    backend = get_backend("pystackreg")
    assert isinstance(backend, PystackregBackend)


def test_get_backend_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        get_backend("unknown")


# ---------------------------------------------------------------------------
# shift_to_tmat unit test
# ---------------------------------------------------------------------------

def test_shift_to_tmat_encodes_shift_correctly() -> None:
    tmat = shift_to_tmat((3.0, -5.0))
    expected = np.array([[1.0, 0.0, -5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    npt.assert_array_equal(tmat, expected)


def test_shift_to_tmat_zero_is_identity() -> None:
    tmat = shift_to_tmat((0.0, 0.0))
    npt.assert_array_equal(tmat, np.eye(3))


# ---------------------------------------------------------------------------
# fit_time – basic interface
# ---------------------------------------------------------------------------

def test_scikit_fit_time_returns_time_model(translated_tyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    model = backend.fit_time(translated_tyx_stack, axes="TYX", method="translation", reference_strategy="first")

    assert model.mode == "time"
    assert model.method == "translation"


def test_scikit_fit_time_transform_shape_tyx(translated_tyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    model = backend.fit_time(translated_tyx_stack, axes="TYX")

    assert isinstance(model.transform, np.ndarray)
    assert model.transform.shape == (translated_tyx_stack.shape[0], 3, 3)


def test_scikit_fit_time_invalid_method_raises(translated_tyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    with pytest.raises(ValueError, match="only supports method='translation'"):
        backend.fit_time(translated_tyx_stack, axes="TYX", method="rigid_body")  # type: ignore[arg-type]


def test_scikit_fit_time_invalid_reference_strategy_raises(translated_tyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    with pytest.raises(ValueError, match="reference_strategy"):
        backend.fit_time(translated_tyx_stack, axes="TYX", reference_strategy="bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# fit_time – reference strategies
# ---------------------------------------------------------------------------

def test_scikit_fit_time_reference_first_frame0_is_identity(translated_tyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    model = backend.fit_time(translated_tyx_stack, axes="TYX", reference_strategy="first")
    npt.assert_array_equal(model.transform[0], np.eye(3))


def test_scikit_fit_time_reference_mean(translated_tyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    model = backend.fit_time(translated_tyx_stack, axes="TYX", reference_strategy="mean")
    assert model.transform.shape == (translated_tyx_stack.shape[0], 3, 3)


def test_scikit_fit_time_reference_previous_produces_cumulative_transforms(
    synthetic_tyx_pairwise: np.ndarray,
) -> None:
    """'previous' strategy must accumulate pairwise shifts so tmat[t] is in frame-0 coordinates."""
    backend = ScikitBackend()
    model = backend.fit_time(synthetic_tyx_pairwise, axes="TYX", reference_strategy="previous")
    tmats = model.transform

    # Frame 0 is always identity
    npt.assert_array_equal(tmats[0], np.eye(3))

    # Cumulative shift for frame 2 must be roughly twice the pairwise shift for frame 1.
    # If pairwise shift is ~-2 in y, cumulative for frame 2 must be ~-4, not ~-2.
    shift_frame1_y = tmats[1, 1, 2]
    shift_frame2_y = tmats[2, 1, 2]
    assert abs(shift_frame2_y) > abs(shift_frame1_y) * 1.5, (
        f"Frame-2 cumulative shift ({shift_frame2_y:.2f}) should be larger than "
        f"frame-1 shift ({shift_frame1_y:.2f}); got raw pairwise instead of cumulative?"
    )


# ---------------------------------------------------------------------------
# apply – shape/dtype preservation
# ---------------------------------------------------------------------------

def test_scikit_apply_preserves_shape_dtype_tyx(translated_tyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    model = backend.fit_time(translated_tyx_stack, axes="TYX")
    result = backend.apply(translated_tyx_stack, axes="TYX", model=model)

    assert result.shape == translated_tyx_stack.shape
    assert result.dtype == translated_tyx_stack.dtype


def test_scikit_apply_preserves_shape_dtype_tcyx(translated_tcyx_stack: np.ndarray) -> None:
    backend = ScikitBackend()
    model = backend.fit_time(translated_tcyx_stack, axes="TCYX", fit_channel=0)
    result = backend.apply(translated_tcyx_stack, axes="TCYX", model=model)

    assert result.shape == translated_tcyx_stack.shape
    assert result.dtype == translated_tcyx_stack.dtype


# ---------------------------------------------------------------------------
# Synthetic sign test – the critical correctness check
# ---------------------------------------------------------------------------

def test_scikit_apply_first_strategy_realigns_known_yx_shift() -> None:
    backend = ScikitBackend()
    base = _blob_frame((48, 48), top=12, left=16)
    known_shift_yx = (5, -3)

    shifted = np.roll(base, shift=known_shift_yx, axis=(0, 1))
    stack = np.stack([base, shifted], axis=0).astype(np.uint16)

    model = backend.fit_time(stack, axes="TYX", reference_strategy="first")
    aligned = backend.apply(stack, axes="TYX", model=model).astype(np.float32)

    npt.assert_allclose(aligned[1], aligned[0], rtol=0.0, atol=1e-3)


def test_scikit_apply_previous_strategy_realigns_cumulative_drift() -> None:
    backend = ScikitBackend()
    base = _blob_frame((48, 48), top=10, left=18)
    pairwise_drift_yx = (2, -1)

    frames = [
        np.roll(base, shift=(0, 0), axis=(0, 1)),
        np.roll(base, shift=pairwise_drift_yx, axis=(0, 1)),
        np.roll(base, shift=(2 * pairwise_drift_yx[0], 2 * pairwise_drift_yx[1]), axis=(0, 1)),
        np.roll(base, shift=(3 * pairwise_drift_yx[0], 3 * pairwise_drift_yx[1]), axis=(0, 1)),
    ]
    stack = np.stack(frames, axis=0).astype(np.uint16)

    model = backend.fit_time(stack, axes="TYX", reference_strategy="previous")
    aligned = backend.apply(stack, axes="TYX", model=model).astype(np.float32)

    for t in range(1, aligned.shape[0]):
        npt.assert_allclose(aligned[t], aligned[0], rtol=0.0, atol=1e-3)

def test_scikit_apply_corrects_known_y_shift(synthetic_tyx_shift3: np.ndarray) -> None:
    """Verify apply undoes the known +3-y shift so peaks align to frame 0."""
    backend = ScikitBackend()
    model = backend.fit_time(synthetic_tyx_shift3, axes="TYX", reference_strategy="first")
    result = backend.apply(synthetic_tyx_shift3, axes="TYX", model=model)

    result_float = result.astype(np.float32)
    # After alignment every frame should peak near the same row as frame 0 (row ~7, centre of blob)
    reference_peak_row = np.unravel_index(np.argmax(result_float[0]), result_float[0].shape)[0]
    for t in range(1, result_float.shape[0]):
        aligned_peak_row = np.unravel_index(np.argmax(result_float[t]), result_float[t].shape)[0]
        assert abs(aligned_peak_row - reference_peak_row) <= 1, (
            f"Frame {t} peak row={aligned_peak_row} is too far from reference row={reference_peak_row}. "
            "Check sign convention in shift_to_tmat / _apply_tyx_substack."
        )


# ---------------------------------------------------------------------------
# channel-wise support lives in dedicated test module
# ---------------------------------------------------------------------------
