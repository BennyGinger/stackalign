from __future__ import annotations

import numpy as np

from stackalign import RegisterModel


def test_register_model_fit_time_then_apply(translated_tcyx_stack: np.ndarray) -> None:
    result = RegisterModel().fit_time(
        translated_tcyx_stack,
        axes="TCYX",
        fit_channel=0,
        method="translation",
        reference_strategy="previous",
    ).apply(translated_tcyx_stack, axes="TCYX")

    assert result.shape == translated_tcyx_stack.shape
    assert result.dtype == translated_tcyx_stack.dtype


def test_register_model_fit_channel_then_apply(translated_tcyx_channel_stack: np.ndarray) -> None:
    result = RegisterModel().fit_channel(
        translated_tcyx_channel_stack,
        axes="TCYX",
        method="translation",
        reference_channel=0,
        reference_frame=0,
    ).apply(translated_tcyx_channel_stack, axes="TCYX")

    assert result.shape == translated_tcyx_channel_stack.shape
    assert result.dtype == translated_tcyx_channel_stack.dtype

