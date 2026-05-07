from __future__ import annotations

import numpy as np
import numpy.testing as npt

from stackalign.backends.pystackreg import PystackregBackend


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


def test_backend_fit_time_empty_t_raises() -> None:
    backend = PystackregBackend()
    empty_tyx = np.zeros((0, 16, 16), dtype=np.float32)

    with np.testing.assert_raises_regex(ValueError, "at least one frame"):
        backend.fit_time(empty_tyx, axes="TYX", method="translation", reference_strategy="first")