from __future__ import annotations

from typing import cast, get_args

import numpy as np
from numpy.typing import NDArray
from pystackreg import StackReg

from stackalign.constants import Method, ReferenceStrategy, DType
from stackalign.backends.models import TransformModel
from stackalign.preparation import ApplyPreparation, FitPreparation

_METHOD_TO_STACKREG = {
    "translation": StackReg.TRANSLATION,
    "rigid_body": StackReg.RIGID_BODY,
    "affine": StackReg.AFFINE,
}

_REFERENCE_STRATEGIES = cast(tuple[ReferenceStrategy, ...], get_args(ReferenceStrategy))
_VALID_REFERENCE_STRATEGIES: frozenset[str] = frozenset(_REFERENCE_STRATEGIES)


class PystackregBackend:
    """Backend that wraps transform fitting and application using pystackreg."""

    def fit_time(self, array: NDArray[np.generic], axes: str, method: Method = "translation", reference_strategy: ReferenceStrategy = "first", fit_channel: int | None = None) -> TransformModel:
        """Fit a time-wise transform model using pystackreg."""
        self._validate_method(method)
        self._validate_reference_strategy(reference_strategy)

        moving_preparation = FitPreparation.for_time(array=array, axes=axes, fit_channel=fit_channel)
        moving_fit = moving_preparation.fit_array
        if moving_fit is None:
            raise RuntimeError("Internal error: fit array was not prepared for fit_time().")

        sr = StackReg(_METHOD_TO_STACKREG[method])
        sr.register_stack(moving_fit, reference=reference_strategy, axis=0)

        return TransformModel(
            mode="time",
            transform=sr,
            reference_channel=None,)

    def fit_channel(self, array: NDArray[np.generic], axes: str, method: Method = "translation", reference_channel: int | None = None, reference_frame: int = 0) -> TransformModel:
        self._validate_method(method)

        preparation = FitPreparation.for_channel(
            array=array,
            axes=axes,
            reference_channel=reference_channel,
            reference_frame=reference_frame,)
        fit_array = preparation.fit_array
        if fit_array is None:
            raise RuntimeError("Internal error: fit array was not prepared for fit_channel().")
        if reference_channel is None:
            raise ValueError("fit_channel requires reference_channel.")

        c_len = fit_array.shape[0]
        transforms: list[StackReg | None] = [None] * c_len
        reference_image = fit_array[reference_channel]

        for channel_index in range(c_len):
            if channel_index == reference_channel:
                continue
            sr = StackReg(_METHOD_TO_STACKREG[method])
            sr.register(reference_image, fit_array[channel_index])
            transforms[channel_index] = sr

        return TransformModel(
            mode="channel",
            transform=transforms,
            reference_channel=reference_channel,)
    
    def apply_time(self, array: NDArray[np.generic], axes: str, model: TransformModel) -> NDArray[np.generic]:
        """Apply a fitted time-wise transform model."""
        if model.mode != "time":
            raise ValueError(f"apply_time requires a time model. Got mode='{model.mode}'.")
        if not isinstance(model.transform, StackReg):
            raise TypeError("apply_time expects model.transform to be a StackReg instance.")

        preparation = ApplyPreparation.for_time(array=array, axes=axes)
        transformed_apply = np.empty(preparation.apply_array.shape, dtype=np.float32)

        for slicer, substack_tyx in preparation.iter_apply_tyx_substacks():
            if substack_tyx.shape[0] == 0:
                transformed_apply[slicer] = substack_tyx
                continue
            transformed_apply[slicer] = model.transform.transform_stack(substack_tyx, axis=0)

        return preparation.restore_apply_output(transformed_apply)

    def apply_channel(self, array: NDArray[np.generic], axes: str, model: TransformModel) -> NDArray[np.generic]:
        if model.mode != "channel":
            raise ValueError(f"apply_channel requires a channel model. Got mode='{model.mode}'.")
        if not isinstance(model.transform, list):
            raise TypeError("apply_channel expects model.transform to be a list of per-channel StackReg objects.")
        if model.reference_channel is None:
            raise ValueError("apply_channel requires a model with reference_channel set.")

        preparation = ApplyPreparation.for_channel(array=array, axes=axes)
        transformed_apply = np.empty(preparation.apply_array.shape, dtype=np.float32)
        c_len = preparation.apply_array.shape[preparation.apply_axes.index("C")]

        if len(model.transform) != c_len:
            raise ValueError(f"Channel model length must match array C length. Got {len(model.transform)} transforms for C={c_len}.")

        for slicer, substack_cyx in preparation.iter_apply_cyx_substacks():
            transformed_substack = np.empty(substack_cyx.shape, dtype=np.float32)
            for channel_index in range(substack_cyx.shape[0]):
                if channel_index == model.reference_channel:
                    transformed_substack[channel_index] = substack_cyx[channel_index]
                    continue

                transform = model.transform[channel_index]
                if transform is None:
                    raise ValueError(f"Missing fitted transform for channel_index={channel_index}.")
                transformed_substack[channel_index] = transform.transform(substack_cyx[channel_index])
            transformed_apply[slicer] = transformed_substack

        return preparation.restore_apply_output(transformed_apply)

    @staticmethod
    def _validate_method(method: str) -> None:
        if method not in _METHOD_TO_STACKREG:
            raise ValueError(f"Unsupported method='{method}'. Supported methods: {sorted(_METHOD_TO_STACKREG)}")

    @staticmethod
    def _validate_reference_strategy(reference_strategy: str) -> None:
        if reference_strategy not in _VALID_REFERENCE_STRATEGIES:
            raise ValueError(f"Unsupported reference_strategy='{reference_strategy}'. Supported: {sorted(_VALID_REFERENCE_STRATEGIES)}")
