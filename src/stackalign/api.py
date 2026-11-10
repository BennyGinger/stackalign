from __future__ import annotations

import numpy as np
from collections.abc import Sequence
from numpy.typing import NDArray
from typing import Self

from stackalign.backends import Backend, get_backend
from stackalign.constants import DType, Method, ReferenceStrategy
from stackalign.backends.models import TransformModel
from stackalign.backends import execution as execution_defaults
from stackalign.backends.execution import CancelCallback, ExecutionMode, ProgressCallback, check_cancelled, validate_execution
from stackalign.preparation import FitPreparation
from stackalign.planes import apply_stack, fit_planes

class RegisterModel:
    """User-facing registration entry point."""

    def __init__(self,
                 backend: str = "pystackreg",
                 *,
                 execution: ExecutionMode | None = None,
                 workers: int | None = None
                 ) -> None:
        """Set the backend and execution policy; auto chooses separately for fit and apply."""
        self.backend = backend.lower()
        self._backend: Backend = get_backend(backend)
        self.execution: ExecutionMode = execution if execution is not None else execution_defaults.EXECUTOR_MODE
        self.workers = workers if workers is not None else execution_defaults.MAX_WORKERS
        validate_execution(self.execution, self.workers)
        self._model: TransformModel | None = None

    @property
    def model(self) -> TransformModel:
        """
        Return a detached model suitable for persistence and later reuse.
        """
        if self._model is None:
            raise RuntimeError("No fitted transform model is available.")
        return TransformModel(self._model.mode, self._model.method,
                              self._model.transform.copy(), self._model.reference_channel)

    def set_model(self, model: TransformModel) -> Self:
        """
        Restore explicit matrices without repeating registration fitting.
        """
        from stackalign.planes import plane_functions
        plane_functions(self.backend, model.method)
        matrices = np.asarray(model.transform, dtype=np.float64)
        if (model.mode not in ("time", "channel") or matrices.ndim != 3
                or matrices.shape[1:] != (3, 3) or not len(matrices)
                or not np.isfinite(matrices).all()):
            raise ValueError("Invalid fitted registration model.")
        if model.mode == "channel" and (model.reference_channel is None
                or not 0 <= model.reference_channel < len(matrices)):
            raise ValueError("Invalid reference channel in fitted registration model.")
        self._model = TransformModel(model.mode, model.method, matrices.copy(), model.reference_channel)
        return self

    def apply_plane(self, array: NDArray[DType], *, frame: int = 0, channel: int = 0) -> NDArray[DType]:
        """
        Apply one fitted transform with the backend's normal dtype restoration.
        """
        from stackalign.planes import apply_plane
        if self._model is None:
            raise RuntimeError("No fitted transform model is available.")
        if self._model.mode == "channel" and channel == self._model.reference_channel:
            return array.copy()
        index = frame if self._model.mode == "time" else channel
        if not 0 <= index < len(self._model.transform):
            raise IndexError("Plane index is outside the fitted transform model.")
        return apply_plane(array, self._model.transform[index], self.backend, self._model.method)

    def fit_time(self,
                 array: NDArray[np.generic],
                 axes: str,
                 method: Method = "translation",
                 reference_strategy: ReferenceStrategy = "first",
                 fit_channel: int | None = None,
                 *,
                 progress: ProgressCallback | None = None,
                 cancelled: CancelCallback | None = None
                 ) -> Self:
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
            Required fitting channel when axes include C. Omit it for a prepared TYX array.

        Returns
        -------
        self
            This RegisterModel instance with fitted time-wise transforms. Call apply() to apply the transforms to compatible arrays.
        """
        check_cancelled(cancelled)
        preparation = FitPreparation.for_time(array, axes, fit_channel)
        assert preparation.fit_array is not None
        return self.fit_time_planes(preparation.fit_array, method=method,
                                    reference_strategy=reference_strategy,
                                    progress=progress, cancelled=cancelled)

    def fit_channel(self,
                    array: NDArray[np.generic],
                    axes: str,
                    method: Method = "translation",
                    reference_channel: int | None = None,
                    reference_frame: int = 0,
                    exclude_channels: Sequence[int] | None = None,
                    *,
                    progress: ProgressCallback | None = None,
                    cancelled: CancelCallback | None = None
                    ) -> Self:
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
            Required channel index used as the alignment reference.
        reference_frame:
            Frame index used as reference for channel fitting. Default is 0.
        exclude_channels:
            Optional channel positions to leave unchanged, without fitting them.

        Returns
        -------
        self
            This RegisterModel instance with fitted channel-wise transforms. Call apply() to apply the transforms to compatible arrays.
        """
        check_cancelled(cancelled)
        preparation = FitPreparation.for_channel(array, axes, reference_channel, reference_frame)
        assert preparation.fit_array is not None and reference_channel is not None
        return self.fit_channel_planes(preparation.fit_array, method=method,
                                       reference_channel=reference_channel,
                                       exclude_channels=exclude_channels,
                                       progress=progress, cancelled=cancelled)

    def fit_time_planes(self,
                        array: NDArray,
                        *,
                        method: Method = "translation",
                        reference_strategy: ReferenceStrategy = "first",
                        progress: ProgressCallback | None = None,
                        cancelled: CancelCallback | None = None
                        ) -> Self:
        """
        Fit already prepared TYX planes, including disk-backed arrays.
        """
        matrices = fit_planes(array, backend=self.backend, method=method,
                              reference=reference_strategy, execution=self.execution,
                              workers=self.workers, progress=progress, cancelled=cancelled)
        self._model = TransformModel("time", method, matrices)
        return self

    def fit_channel_planes(self,
                           array: NDArray,
                           *,
                           reference_channel: int,
                           method: Method = "translation",
                           exclude_channels: Sequence[int] | None = None,
                           progress: ProgressCallback | None = None,
                           cancelled: CancelCallback | None = None
                           ) -> Self:
        """
        Fit already prepared CYX planes at one frame, with excluded channels.
        """
        matrices = fit_planes(array, backend=self.backend, method=method,
                              reference=reference_channel, excluded=exclude_channels or (),
                              execution=self.execution, workers=self.workers,
                              progress=progress, cancelled=cancelled)
        self._model = TransformModel("channel", method, matrices, reference_channel)
        return self

    def apply(self,
              array: NDArray[DType],
              axes: str,
              *,
              out: NDArray[DType] | None = None,
              progress: ProgressCallback | None = None,
              cancelled: CancelCallback | None = None
              ) -> NDArray[DType]:
        """
        Apply the currently fitted transforms to a compatible array.

        Supply ``out`` to write into an existing array or memmap. Progress and
        cancellation callbacks run in the calling thread; partial output is
        left with the caller if work fails or is cancelled.

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
        return apply_stack(array, axes, self._model, backend=self.backend,
                           execution=self.execution, workers=self.workers, out=out,
                           progress=progress, cancelled=cancelled)



if __name__ == "__main__":
    from pathlib import Path
    from time import time
    from scope_io import ScopeIO
    from tifffile import imwrite

    path = Path("/media/ben/Analysis/Python/Images/NeutrophilTrackingTest/dia/c1133-MaxIP_s1/fits_array.tif")

    reader = ScopeIO.from_path(path)
    array = reader.get_array().array
    if isinstance(array, list):
        array = array[0]
    print(f"Original array shape: {array.shape}, dtype: {array.dtype} and axes: {reader.axes}")


    # Time-wise example usage
    bf = array[:,2,:,:]  # take one channel for testing
    bf_axis = reader.axes[0].replace("C", "")  # pretend this is a ZYX stack for testing
    print(f"Input array shape: {bf.shape}, dtype: {bf.dtype} and axes: {bf_axis}")
    register = RegisterModel(backend="pystackreg")
    for _ in range(5):  # run multiple times to get an idea of runtime variability
        t0 = time()
        register.fit_time(array=bf, axes=bf_axis,
                        method="translation",
                        reference_strategy="previous")
        print(f"Fitting completed in {time() - t0:.2f} seconds.")
        t1 = time()
        transformed = register.apply(array=bf, axes=bf_axis)
        print(f"Applying transforms completed in {time() - t1:.2f} seconds.")
        t2 = time()
        # imwrite(path.with_name(path.stem + "_registered.tif"), transformed)
        # print(f"Saving registered array completed in {time() - t2:.2f} seconds.")
        print(f"Registration completed in {time() - t0:.2f} seconds.")

    # # Channel-wise example usage
    # chan_arr = array[:, :2, :, :]
    # print(f"Input array shape: {chan_arr.shape}, dtype: {chan_arr.dtype}")
    # t0 = time()
    # register = RegisterModel(backend="cv2")
    # register.fit_channel(array=chan_arr, axes=reader.axes[0],
    #                      method="translation",
    #                      reference_channel=1,
    #                      reference_frame=0)
    # print(f"Fitting completed in {time() - t0:.2f} seconds.")
    # t1 = time()
    # transformed = register.apply(array=chan_arr, axes=reader.axes[0])
    # print(f"Applying transforms completed in {time() - t1:.2f} seconds.")
    # t2 = time()
    # imwrite(path.with_name(path.stem + "_channel_registered.tif"), transformed)
    # print(f"Saving registered array completed in {time() - t2:.2f} seconds.")
    # print(f"Channel registration completed in {time() - t0:.2f} seconds.")
