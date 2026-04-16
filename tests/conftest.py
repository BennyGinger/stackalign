from __future__ import annotations

import numpy as np
import pytest


def _bright_square(shape: tuple[int, int], top: int, left: int, size: int = 4, value: int = 1000) -> np.ndarray:
    frame = np.zeros(shape, dtype=np.uint16)
    frame[top : top + size, left : left + size] = value
    return frame


@pytest.fixture
def translated_tyx_stack() -> np.ndarray:
    frames = [
        _bright_square((20, 20), 3, 3),
        _bright_square((20, 20), 4, 5),
        _bright_square((20, 20), 5, 7),
        _bright_square((20, 20), 6, 8),
    ]
    return np.stack(frames, axis=0)


@pytest.fixture
def translated_tcyx_stack(translated_tyx_stack: np.ndarray) -> np.ndarray:
    return np.stack([translated_tyx_stack, translated_tyx_stack * 2], axis=1)


@pytest.fixture
def translated_tczyx_stack(translated_tyx_stack: np.ndarray) -> np.ndarray:
    z0 = translated_tyx_stack
    z1 = translated_tyx_stack * 2
    channel0 = np.stack([z0, z1], axis=1)
    channel1 = channel0 * 3
    return np.stack([channel0, channel1], axis=1)


def _channel_frame(top: int, left: int, value: int = 1000) -> np.ndarray:
    return _bright_square((24, 24), top=top, left=left, size=5, value=value)


@pytest.fixture
def translated_cyx_stack() -> np.ndarray:
    return np.stack(
        [
            _channel_frame(6, 6, value=1000),
            _channel_frame(8, 9, value=1000),
            _channel_frame(4, 10, value=1000),
        ],
        axis=0,
    )


@pytest.fixture
def translated_tcyx_channel_stack() -> np.ndarray:
    frames: list[np.ndarray] = []
    for time_index, (top, left) in enumerate([(6, 6), (8, 7), (9, 9)]):
        reference = _channel_frame(top, left, value=1000 + 50 * time_index)
        shifted_a = _channel_frame(top + 2, left + 3, value=1000 + 50 * time_index)
        shifted_b = _channel_frame(top - 2, left + 4, value=1000 + 50 * time_index)
        frames.append(np.stack([reference, shifted_a, shifted_b], axis=0))
    return np.stack(frames, axis=0).astype(np.uint16)


@pytest.fixture
def translated_tczyx_channel_stack() -> np.ndarray:
    frames: list[np.ndarray] = []
    for time_index, (top, left) in enumerate([(6, 6), (8, 7)]):
        z_planes: list[np.ndarray] = []
        for z_index, value_scale in enumerate([1, 2]):
            reference = _channel_frame(top + z_index, left, value=(1000 + 100 * time_index) * value_scale)
            shifted_a = _channel_frame(top + z_index + 2, left + 3, value=(1000 + 100 * time_index) * value_scale)
            shifted_b = _channel_frame(top + z_index - 2, left + 4, value=(1000 + 100 * time_index) * value_scale)
            z_planes.append(np.stack([reference, shifted_a, shifted_b], axis=0))
        frames.append(np.stack(z_planes, axis=1))
    return np.stack(frames, axis=0).astype(np.uint16)
