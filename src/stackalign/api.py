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
        return self._backend.apply(array=array, axes=axes, model=self._model)
    
if __name__ == "__main__":
    from pathlib import Path
    from time import time
    from fits_io import FitsIO
    from tifffile import imwrite
    
    path = Path("/media/ben/Analysis/Python/Images/NeutrophilTrackingTest/dia/c1133-MaxIP.nd2 - c1133-MaxIP.nd2 (series 1).tif")
    
    reader = FitsIO.from_path(path)
    array = reader.get_array()
    
    if isinstance(array, list):
        array = array[0]
    
    
    # # Time-wise example usage
    # bf = array[:,3,:,:]  # take one channel for testing
    # bf_axis = reader.axes[0].replace("C", "")  # pretend this is a ZYX stack for testing 
    # print(f"Input array shape: {bf.shape}, dtype: {bf.dtype} and axes: {bf_axis}")
    # t0 = time()
    # register = RegisterModel(backend="cv2")
    # register.fit_time(array=bf, axes=bf_axis, 
    #                   method="affine", 
    #                   reference_strategy="previous")
    # print(f"Fitting completed in {time() - t0:.2f} seconds.")
    # t1 = time()
    # transformed = register.apply(array=bf, axes=bf_axis)
    # print(f"Applying transforms completed in {time() - t1:.2f} seconds.")
    # t2 = time()
    # imwrite(path.with_name(path.stem + "_registered.tif"), transformed)
    # print(f"Saving registered array completed in {time() - t2:.2f} seconds.")
    # print(f"Registration completed in {time() - t0:.2f} seconds.")
    
    # Channel-wise example usage
    chan_arr = array[:, :2, :, :]  # take first 4 channels for testing
    print(f"Input array shape: {chan_arr.shape}, dtype: {chan_arr.dtype}")
    t0 = time()
    register = RegisterModel(backend="pystackreg")
    register.fit_channel(array=chan_arr, axes=reader.axes[0],
                         method="translation",
                         reference_channel=1,
                         reference_frame=0)
    print(f"Fitting completed in {time() - t0:.2f} seconds.")
    t1 = time()
    transformed = register.apply(array=chan_arr, axes=reader.axes[0])
    print(f"Applying transforms completed in {time() - t1:.2f} seconds.")
    t2 = time()
    imwrite(path.with_name(path.stem + "_channel_registered.tif"), transformed)
    print(f"Saving registered array completed in {time() - t2:.2f} seconds.")
    print(f"Channel registration completed in {time() - t0:.2f} seconds.")