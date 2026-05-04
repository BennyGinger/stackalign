from __future__ import annotations

import numpy as np
import numpy.testing as npt

from stackalign.backends.pystackreg import PystackregBackend


def _center_of_mass(image: np.ndarray) -> tuple[float, float]:
    weights = image.astype(np.float64)
    total = float(weights.sum())
    if total == 0.0:
        raise ValueError("center_of_mass requires non-empty signal.")
    yy, xx = np.indices(weights.shape)
    return float((yy * weights).sum() / total), float((xx * weights).sum() / total)


def test_backend_time_fit_and_apply_roundtrip_shape_dtype_tyx(translated_tyx_stack: np.ndarray) -> None:
    backend = PystackregBackend()

    model = backend.fit_time(translated_tyx_stack, axes="TYX", method="translation", reference_strategy="previous")
    result = backend.apply(translated_tyx_stack, axes="TYX", model=model)

    assert model.mode == "time"
    assert isinstance(model.transform, np.ndarray)
    assert model.transform.shape[0] == translated_tyx_stack.shape[0]
    assert model.transform.shape[1:] == (3, 3)
    assert result.shape == translated_tyx_stack.shape
    assert result.dtype == translated_tyx_stack.dtype


def test_backend_apply_on_tcyx_applies_same_time_transform_to_all_channels(translated_tcyx_stack: np.ndarray) -> None:
    backend = PystackregBackend()

    model = backend.fit_time(translated_tcyx_stack, axes="TCYX", fit_channel=0, method="translation", reference_strategy="previous")
    result = backend.apply(translated_tcyx_stack, axes="TCYX", model=model)

    assert result.shape == translated_tcyx_stack.shape
    assert result.dtype == translated_tcyx_stack.dtype
    npt.assert_allclose(result[:, 1].astype(np.float32), result[:, 0].astype(np.float32) * 2.0, atol=1.0)


def test_backend_apply_on_tczyx_preserves_c_and_z(translated_tczyx_stack: np.ndarray) -> None:
    backend = PystackregBackend()

    model = backend.fit_time(translated_tczyx_stack, axes="TCZYX", fit_channel=0, method="translation", reference_strategy="previous")
    result = backend.apply(translated_tczyx_stack, axes="TCZYX", model=model)

    assert result.shape == translated_tczyx_stack.shape
    assert result.dtype == translated_tczyx_stack.dtype
    npt.assert_allclose(result[:, 0, 1].astype(np.float32), result[:, 0, 0].astype(np.float32) * 2.0, atol=1.0)
    npt.assert_allclose(result[:, 1, 0].astype(np.float32), result[:, 0, 0].astype(np.float32) * 3.0, atol=1.0)


def test_backend_fit_channel_on_cyx_returns_one_transform_per_channel(translated_cyx_stack: np.ndarray) -> None:
    backend = PystackregBackend()

    model = backend.fit_channel(translated_cyx_stack, axes="CYX", method="translation", reference_channel=0, reference_frame=0)

    assert model.mode == "channel"
    assert isinstance(model.transform, np.ndarray)
    assert model.transform.shape[0] == translated_cyx_stack.shape[0]
    assert model.transform.shape[1:] == (3, 3)
    assert model.reference_channel == 0
    npt.assert_allclose(model.transform[0], np.eye(3, dtype=np.float64), atol=1e-8)


def test_backend_fit_channel_on_tcyx_uses_reference_frame_zero(translated_tcyx_channel_stack: np.ndarray) -> None:
    backend = PystackregBackend()

    model = backend.fit_channel(translated_tcyx_channel_stack, axes="TCYX", method="translation", reference_channel=0, reference_frame=0)
    result = backend.apply(translated_tcyx_channel_stack, axes="TCYX", model=model)

    assert result.shape == translated_tcyx_channel_stack.shape
    assert result.dtype == translated_tcyx_channel_stack.dtype
    npt.assert_allclose(result[:, 0].astype(np.float32), translated_tcyx_channel_stack[:, 0].astype(np.float32), atol=1.0)
    for time_index in range(result.shape[0]):
        reference_center = _center_of_mass(result[time_index, 0])
        for channel_index in (1, 2):
            channel_center = _center_of_mass(result[time_index, channel_index])
            assert abs(channel_center[0] - reference_center[0]) < 0.25
            assert abs(channel_center[1] - reference_center[1]) < 0.25


def test_backend_fit_channel_on_tczyx_uses_max_projection_for_fit_and_preserves_dims(translated_tczyx_channel_stack: np.ndarray) -> None:
    backend = PystackregBackend()

    model = backend.fit_channel(translated_tczyx_channel_stack, axes="TCZYX", method="translation", reference_channel=0, reference_frame=0)
    result = backend.apply(translated_tczyx_channel_stack, axes="TCZYX", model=model)

    assert result.shape == translated_tczyx_channel_stack.shape
    assert result.dtype == translated_tczyx_channel_stack.dtype
    npt.assert_allclose(result[:, 0].astype(np.float32), translated_tczyx_channel_stack[:, 0].astype(np.float32), atol=1.0)
    for time_index in range(result.shape[0]):
        for z_index in range(result.shape[2]):
            reference_center = _center_of_mass(result[time_index, 0, z_index])
            for channel_index in (1, 2):
                channel_center = _center_of_mass(result[time_index, channel_index, z_index])
                assert abs(channel_center[0] - reference_center[0]) < 0.25
                assert abs(channel_center[1] - reference_center[1]) < 0.25


def test_backend_apply_channel_preserves_original_axis_order(translated_tcyx_channel_stack: np.ndarray) -> None:
    backend = PystackregBackend()
    array = np.transpose(translated_tcyx_channel_stack, (2, 3, 0, 1))

    model = backend.fit_channel(translated_tcyx_channel_stack, axes="TCYX", method="translation", reference_channel=0, reference_frame=0)
    result = backend.apply(array, axes="YXTC", model=model)

    assert result.shape == array.shape
    assert result.dtype == array.dtype


def test_backend_fit_channel_missing_c_axis_raises(translated_tyx_stack: np.ndarray) -> None:
    backend = PystackregBackend()

    with np.testing.assert_raises_regex(ValueError, "axis 'C'"):
        backend.fit_channel(translated_tyx_stack, axes="TYX", reference_channel=0)


def test_backend_fit_channel_missing_reference_channel_raises(translated_cyx_stack: np.ndarray) -> None:
    backend = PystackregBackend()

    with np.testing.assert_raises_regex(ValueError, "reference_channel"):
        backend.fit_channel(translated_cyx_stack, axes="CYX", reference_channel=None)


def test_backend_fit_channel_invalid_reference_frame_raises(translated_cyx_stack: np.ndarray) -> None:
    backend = PystackregBackend()

    with np.testing.assert_raises_regex(ValueError, "reference_frame"):
        backend.fit_channel(translated_cyx_stack, axes="CYX", reference_channel=0, reference_frame=1)


def test_backend_fit_time_empty_t_raises() -> None:
    backend = PystackregBackend()
    empty_tyx = np.zeros((0, 16, 16), dtype=np.float32)

    with np.testing.assert_raises_regex(ValueError, "at least one frame"):
        backend.fit_time(empty_tyx, axes="TYX", method="translation", reference_strategy="first")
