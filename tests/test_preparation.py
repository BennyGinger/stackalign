from __future__ import annotations

import numpy as np
import pytest

from stackalign.preparation import ApplyPreparation, FitPreparation


def test_fit_preparation_tyx_keeps_tyx(translated_tyx_stack: np.ndarray) -> None:
    preparation = FitPreparation.for_time(translated_tyx_stack, axes="TYX")

    assert preparation.fit_array is not None
    assert preparation.fit_array.shape == translated_tyx_stack.shape
    assert preparation.fit_axes == "TYX"


def test_fit_preparation_tcyx_selects_fit_channel(translated_tcyx_stack: np.ndarray) -> None:
    preparation = FitPreparation.for_time(translated_tcyx_stack, axes="TCYX", fit_channel=1)

    assert preparation.fit_array is not None
    assert preparation.fit_axes == "TYX"
    np.testing.assert_array_equal(preparation.fit_array, translated_tcyx_stack[:, 1].astype(np.float32))


def test_fit_preparation_tcyx_missing_fit_channel_raises(translated_tcyx_stack: np.ndarray) -> None:
    with pytest.raises(ValueError, match="fit_channel"):
        FitPreparation.for_time(translated_tcyx_stack, axes="TCYX")


def test_fit_preparation_tzyx_projects_z_to_tyx(translated_tyx_stack: np.ndarray) -> None:
    array = np.stack([translated_tyx_stack, translated_tyx_stack * 3], axis=1)

    preparation = FitPreparation.for_time(array, axes="TZYX")

    assert preparation.fit_array is not None
    assert preparation.fit_axes == "TYX"
    np.testing.assert_array_equal(preparation.fit_array, array.max(axis=1).astype(np.float32))


@pytest.mark.parametrize(
    ("axes", "array"),
    [
        ("YX", np.pad(np.full((4, 4), 1000, dtype=np.uint16), ((2, 6), (3, 3)))),
        ("CYX", np.stack([np.pad(np.full((4, 4), 1000, dtype=np.uint16), ((2, 6), (3, 3))), np.pad(np.full((4, 4), 1000, dtype=np.uint16), ((4, 4), (5, 1)))], axis=0)),
    ],
)
def test_apply_preparation_adds_synthetic_t_when_missing(axes: str, array: np.ndarray) -> None:
    preparation = ApplyPreparation.for_time(array, axes=axes)

    assert preparation.synthetic_time_axis is True
    assert preparation.apply_axes.startswith("T")
    assert preparation.apply_array.shape[0] == 1

    restored = preparation.restore_apply_output(preparation.apply_array.astype(np.float32))

    assert restored.shape == array.shape
    assert restored.dtype == array.dtype
    np.testing.assert_array_equal(restored, array)


def test_apply_preparation_restores_original_axes_order(translated_tyx_stack: np.ndarray) -> None:
    array = np.transpose(translated_tyx_stack, (1, 2, 0))

    preparation = ApplyPreparation.for_time(array, axes="YXT")
    transformed = preparation.apply_array.astype(np.float32) + 1.0
    restored = preparation.restore_apply_output(transformed)

    assert restored.shape == array.shape
    assert restored.dtype == array.dtype
    np.testing.assert_array_equal(restored, array + 1)


@pytest.mark.parametrize(
    ("array", "axes", "message"),
    [
        (np.zeros((2, 3, 4), dtype=np.uint16), "TYXC", "length"),
        (np.zeros((2, 3, 4), dtype=np.uint16), "TXX", "duplicates"),
        (np.zeros((2, 3, 4), dtype=np.uint16), "TCZ", "axis 'X'"),
        (np.zeros((2, 3, 4), dtype=np.uint16), "TXZ", "axis 'Y'"),
    ],
)
def test_invalid_axes_validation_failures(array: np.ndarray, axes: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ApplyPreparation.for_time(array, axes=axes)


def test_fit_preparation_missing_t_for_fit_time_raises() -> None:
    array = np.zeros((3, 6, 6), dtype=np.uint16)

    with pytest.raises(ValueError, match="axis 'T'"):
        FitPreparation.for_time(array, axes="ZYX")


def test_fit_channel_out_of_range_raises(translated_tcyx_stack: np.ndarray) -> None:
    with pytest.raises(ValueError, match="out of range"):
        FitPreparation.for_time(translated_tcyx_stack, axes="TCYX", fit_channel=5)


def test_fit_channel_without_c_axis_raises(translated_tyx_stack: np.ndarray) -> None:
    with pytest.raises(ValueError, match="does not include C"):
        FitPreparation.for_time(translated_tyx_stack, axes="TYX", fit_channel=0)


def test_fit_preparation_channel_cyx_keeps_cyx(translated_cyx_stack: np.ndarray) -> None:
    preparation = FitPreparation.for_channel(translated_cyx_stack, axes="CYX", reference_channel=0, reference_frame=0)

    assert preparation.fit_array is not None
    assert preparation.fit_axes == "CYX"
    np.testing.assert_array_equal(preparation.fit_array, translated_cyx_stack.astype(np.float32))


def test_fit_preparation_channel_tcyx_selects_reference_frame(translated_tcyx_channel_stack: np.ndarray) -> None:
    preparation = FitPreparation.for_channel(translated_tcyx_channel_stack, axes="TCYX", reference_channel=0, reference_frame=0)

    assert preparation.fit_array is not None
    assert preparation.fit_axes == "CYX"
    np.testing.assert_array_equal(preparation.fit_array, translated_tcyx_channel_stack[0].astype(np.float32))


def test_fit_preparation_channel_tczyx_projects_z_max(translated_tczyx_channel_stack: np.ndarray) -> None:
    preparation = FitPreparation.for_channel(translated_tczyx_channel_stack, axes="TCZYX", reference_channel=0, reference_frame=0)

    assert preparation.fit_array is not None
    assert preparation.fit_axes == "CYX"
    np.testing.assert_array_equal(preparation.fit_array, translated_tczyx_channel_stack[0].max(axis=1).astype(np.float32))


def test_apply_preparation_channel_restores_original_axes_order(translated_cyx_stack: np.ndarray) -> None:
    array = np.transpose(translated_cyx_stack, (1, 2, 0))

    preparation = ApplyPreparation.for_channel(array, axes="YXC")
    transformed = preparation.apply_array.astype(np.float32) + 1.0
    restored = preparation.restore_apply_output(transformed)

    assert restored.shape == array.shape
    assert restored.dtype == array.dtype
    np.testing.assert_array_equal(restored, array + 1)


def test_fit_preparation_channel_missing_c_axis_raises(translated_tyx_stack: np.ndarray) -> None:
    with pytest.raises(ValueError, match="axis 'C'"):
        FitPreparation.for_channel(translated_tyx_stack, axes="TYX", reference_channel=0)


def test_fit_preparation_channel_missing_reference_channel_raises(translated_cyx_stack: np.ndarray) -> None:
    with pytest.raises(ValueError, match="reference_channel"):
        FitPreparation.for_channel(translated_cyx_stack, axes="CYX", reference_channel=None)


def test_fit_preparation_channel_reference_channel_out_of_range_raises(translated_cyx_stack: np.ndarray) -> None:
    with pytest.raises(ValueError, match="out of range"):
        FitPreparation.for_channel(translated_cyx_stack, axes="CYX", reference_channel=10)


def test_fit_preparation_channel_invalid_reference_frame_raises_without_t(translated_cyx_stack: np.ndarray) -> None:
    with pytest.raises(ValueError, match="reference_frame"):
        FitPreparation.for_channel(translated_cyx_stack, axes="CYX", reference_channel=0, reference_frame=1)
