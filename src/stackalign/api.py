from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Self

from stackalign.backends import Backend, get_backend
from stackalign.constants import Method, ReferenceStrategy
from stackalign.backends.models import TransformModel


class RegisterModel:
    """User-facing registration entry point."""

    def __init__(self, backend: str = "pystackreg") -> None:
        self._backend: Backend = get_backend(backend)
        self._model: TransformModel | None = None

    def fit_time(self, array: NDArray[np.generic], axes: str, method: Method = "translation", reference_strategy: ReferenceStrategy = "first", fit_channel: int | None = None) -> Self:
        """Fit time-wise transforms and store them on this RegisterModel.

        Parameters
        ----------
        array:
            Input moving array.
        axes:
            Axes of array. Any order is accepted as long as labels are from T/C/Z/Y/X and include Y and X.
        method:
            Registration method: translation, rigid_body, affine.
        reference_strategy:
            Reference policy for time fitting: first, previous, mean.
        fit_channel:
            Channel index used for fitting when working with TCYX data.
        
        Returns
        -------
        self
            This RegisterModel instance with fitted time-wise transforms. Call apply() to apply the transforms to compatible arrays.
        """
        self._model = self._backend.fit_time(
            array=array,
            axes=axes,
            method=method,
            reference_strategy=reference_strategy,
            fit_channel=fit_channel,)
        return self

    def fit_channel(self, array: NDArray[np.generic], axes: str, method: Method = "translation", reference_channel: int | None = None, reference_frame: int = 0) -> Self:
        """
        Fit channel-wise transforms and store them on this RegisterModel.
        
        Parameters
        ----------
        array:
            Input moving array.
        axes:
            Axes of array. Any order is accepted as long as labels are from T/C/Z/Y/X and include Y and X.
        method:
            Registration method: translation, rigid_body, affine.
        reference_channel:
            Channel index used as reference for channel fitting. If None, the reference channel will be determined automatically as the channel with the highest average intensity across frames.
        reference_frame:
            Frame index used as reference for channel fitting. Default is 0.
        
        Returns
        -------
        self
            This RegisterModel instance with fitted channel-wise transforms. Call apply() to apply the transforms to compatible arrays.
        """
        self._model = self._backend.fit_channel(
            array=array,
            axes=axes,
            method=method,
            reference_channel=reference_channel,
            reference_frame=reference_frame)
        return self

    def apply(self, array: NDArray[np.generic], axes: str) -> NDArray[np.generic]:
        """
        Apply the currently fitted transforms to a compatible array.
        
        Parameters
        ----------
        array:
            Input array to transform. Must be compatible with the fitted model (e.g. if fit_time was called on TCYX data, this should also be TCYX).
        axes:
            Axes of array. Any order is accepted as long as labels are from T/C/Z/Y/X and include Y and X.
        
        Returns
        -------
        NDArray[np.generic]
            Transformed array with the same shape and dtype as the input array.
        """
        if self._model is None:
            raise RuntimeError("No fitted transform model is available. Call fit_time() or fit_channel() first.")

        if self._model.mode == "time":
            return self._backend.apply_time(array=array, axes=axes, model=self._model)

        if self._model.mode == "channel":
            return self._backend.apply_channel(array=array, axes=axes, model=self._model)

        raise RuntimeError(f"Unsupported transform mode '{self._model.mode}'.")
    
    