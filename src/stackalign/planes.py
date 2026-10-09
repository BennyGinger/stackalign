"""Serial plane operations shared by streaming registration clients.

These use the same backend routines as whole-stack registration, without
launching a process pool for each plane in an interactive preview.
"""

from collections.abc import Callable, Collection
from functools import partial

import numpy as np
from numpy.typing import NDArray

from stackalign.backends.protocol import FrameApplyFn, FrameFitFn
from stackalign.constants import Method
from stackalign.preparation import ApplyPreparation


def plane_functions(backend: str, method: Method) -> tuple[FrameFitFn, FrameApplyFn]:
    if backend == "pystackreg":
        from stackalign.backends.pystackreg.utils import (
            apply_frame_tmat_task, register_frame_to_reference_task, validate_method)
        validate_method(method)
        return (partial(register_frame_to_reference_task, method=method),
                partial(apply_frame_tmat_task, method=method))
    if backend == "cv2":
        from stackalign.backends.cv2.time_wise import _apply_frame_task, _fit_frame_to_reference_task
        from stackalign.backends.cv2.utils import validate_method
        validate_method(method)
        return (partial(_fit_frame_to_reference_task, method=method),
                partial(_apply_frame_task, method=method))
    if backend == "scikit":
        from stackalign.backends.scikit.time_wise import _apply_frame_task, _fit_frame_to_reference_task
        from stackalign.backends.scikit.utils import validate_method
        validate_method(method)
        return _fit_frame_to_reference_task, _apply_frame_task
    raise ValueError(f"Unsupported backend {backend!r}.")


def apply_plane(array: NDArray, matrix: NDArray, backend: str, method: Method) -> NDArray:
    if array.ndim != 2:
        raise ValueError("apply_plane requires a YX image.")
    _, apply = plane_functions(backend, method)
    _, transformed = apply(0, np.asarray(array, dtype=np.float32), matrix)
    return ApplyPreparation._restore_dtype(transformed, array.dtype)


def fit_planes(array: NDArray, *, backend: str, method: Method,
               reference: str | int, progress: Callable[[int, int], None] | None = None,
               cancelled: Callable[[], bool] | None = None,
               excluded: Collection[int] = ()) -> NDArray[np.float64]:
    """Fit TYX/CYX planes with bounded working memory and cooperative cancellation."""
    from stackalign.backends.transforms import identity_tmats, validate_reference_strategy
    if array.ndim != 3 or len(array) == 0:
        raise ValueError("Fitting requires a nonempty TYX or CYX array.")
    if isinstance(reference, str):
        validate_reference_strategy(reference)
    elif not 0 <= reference < len(array):
        raise ValueError("Reference channel is outside the fitting array.")
    if any(not 0 <= index < len(array) for index in excluded) or reference in excluded:
        raise ValueError("Excluded channels must be valid and cannot include the reference channel.")
    fit, _ = plane_functions(backend, method)
    fixed = (array.mean(axis=0, dtype=np.float32) if reference == "mean"
             else array[reference if isinstance(reference, int) else 0])
    matrices = identity_tmats(len(array))
    for index in range(len(array)):
        if cancelled is not None and cancelled():
            raise InterruptedError("Registration preview cancelled.")
        if (reference == "previous" and index == 0) or index == reference or index in excluded:
            pass
        else:
            target = array[index - 1] if reference == "previous" else fixed
            _, matrix = fit(index, np.asarray(target, dtype=np.float32),
                            np.asarray(array[index], dtype=np.float32))
            matrices[index] = (matrix @ matrices[index - 1]
                               if reference == "previous" else matrix)
        if progress is not None:
            progress(index + 1, len(array))
    return matrices
