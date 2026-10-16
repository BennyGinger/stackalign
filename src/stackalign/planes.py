"""Shared fitting and plane-wise application for array and streaming clients."""

from collections.abc import Callable, Collection
from functools import partial

import numpy as np
from numpy.typing import NDArray

from stackalign.backends.protocol import FrameApplyFn, FrameFitFn
from stackalign.backends.execution import CancelCallback, ExecutionMode, ProgressCallback, check_cancelled, run_tasks, validate_execution
from stackalign.backends.models import TransformModel
from stackalign.axes import canonical_axes, move_to_axes, normalize_axes, require_axes_member
from stackalign.constants import Method
from stackalign.preparation import ApplyPreparation


def plane_functions(backend: str, method: Method) -> tuple[FrameFitFn, FrameApplyFn]:
    """
    Return the fitting and application kernels for one backend and method.
    """
    if backend == "pystackreg":
        from stackalign.backends.pystackreg.utils import apply_frame_tmat_task, register_frame_to_reference_task, validate_method
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
    """
    Transform one YX image and restore its original dtype.
    """
    if array.ndim != 2:
        raise ValueError("apply_plane requires a YX image.")
    _, apply = plane_functions(backend, method)
    _, transformed = apply(0, np.asarray(array, dtype=np.float32), matrix)
    return ApplyPreparation._restore_dtype(transformed, array.dtype)


def fit_planes(array: NDArray,
               *,
               backend: str,
               method: Method,
               reference: str | int,
               progress: Callable[[int, int], None] | None = None,
               cancelled: Callable[[], bool] | None = None,
               excluded: Collection[int] = (),
               execution: ExecutionMode | None = None,
               workers: int | None = None
               ) -> NDArray[np.float64]:
    """
    Fit TYX/CYX planes with bounded working memory and cooperative cancellation.
    """
    from stackalign.backends.transforms import accumulate_pairwise_tmats, identity_tmats, validate_reference_strategy
    if array.ndim != 3 or len(array) == 0:
        raise ValueError("Fitting requires a nonempty TYX or CYX array.")
    if isinstance(reference, str):
        validate_reference_strategy(reference)
    elif not 0 <= reference < len(array):
        raise ValueError("Reference channel is outside the fitting array.")
    if any(not 0 <= index < len(array) for index in excluded) or reference in excluded:
        raise ValueError("Excluded channels must be valid and cannot include the reference channel.")
    if execution is not None:
        validate_execution(execution, workers)
    check_cancelled(cancelled)
    fit, _ = plane_functions(backend, method)
    fixed = (array.mean(axis=0, dtype=np.float32) if reference == "mean"
             else array[reference if isinstance(reference, int) else 0])
    matrices = identity_tmats(len(array))
    indices = [index for index in range(len(array))
               if not ((reference == "previous" and index == 0)
                       or index == reference or index in excluded)]
    tasks = ((index, np.asarray(array[index - 1] if reference == "previous" else fixed,
                               dtype=np.float32), np.asarray(array[index], dtype=np.float32))
             for index in indices)
    for index, matrix in run_tasks(fit, tasks, total=len(indices), execution=execution,
                                   backend=backend, method=method, operation="fit",
                                   plane_pixels=int(np.prod(array.shape[-2:])),
                                   workers=workers, progress=progress, cancelled=cancelled):
        matrices[index] = matrix
    return accumulate_pairwise_tmats(matrices) if reference == "previous" else matrices


def _apply_plane_task(index: int,
                      plane: NDArray,
                      matrix: NDArray,
                      backend: str,
                      method: Method
                      ) -> tuple[int, NDArray]:
    """
    Transform one image and return it with its destination index.
    """
    return index, apply_plane(plane, matrix, backend, method)


def apply_stack(array: NDArray,
                axes: str,
                model: TransformModel,
                *,
                backend: str,
                execution: ExecutionMode | None = None,
                workers: int | None = None,
                out: NDArray | None = None,
                progress: ProgressCallback | None = None,
                cancelled: CancelCallback | None = None
                ) -> NDArray:
    """
    Apply through one pool, restoring dtype per plane instead of copying a full stack.

    ``out`` may be a writable memmap. It can contain partial results after a
    failure or cancellation; callers own its publication and cleanup.
    """
    source = np.asarray(array)
    if execution is not None:
        validate_execution(execution, workers)
    source_axes = normalize_axes(source, axes)
    axis = "T" if model.mode == "time" else "C"
    require_axes_member(source_axes, axis, "apply")
    matrices = np.asarray(model.transform, dtype=np.float64)
    count = source.shape[source_axes.index(axis)]
    if matrices.shape != (count, 3, 3) or not np.isfinite(matrices).all():
        raise ValueError(f"{model.mode.title()} model length must match array {axis} length "
                         f"and contain finite 3x3 transforms; expected ({count}, 3, 3).")
    if model.mode == "channel" and (model.reference_channel is None
                                   or not 0 <= model.reference_channel < count):
        raise ValueError("Channel model requires a valid reference channel.")
    plane_functions(backend, model.method)
    check_cancelled(cancelled)
    output = np.empty_like(source) if out is None else out
    if (output.shape != source.shape or output.dtype != source.dtype
            or not output.flags.writeable):
        raise ValueError("out must be writable and match the input shape and dtype.")
    if np.may_share_memory(source, output):
        raise ValueError("out must not overlap the input array.")
    canonical = canonical_axes(source_axes)
    input_view = move_to_axes(source, source_axes, canonical)
    output_view = move_to_axes(output, source_axes, canonical)
    navigation_shape = input_view.shape[:-2]
    model_axis = canonical.index(axis)
    total = int(np.prod(navigation_shape))
    if model.mode == "channel":
        for position in np.ndindex(navigation_shape):
            if position[model_axis] == model.reference_channel:
                check_cancelled(cancelled)
                output_view[position] = input_view[position]
        total -= total // count

    tasks = ((index, input_view[position], matrices[position[model_axis]])
             for index, position in enumerate(np.ndindex(navigation_shape))
             if model.mode != "channel" or position[model_axis] != model.reference_channel)
    apply = partial(_apply_plane_task, backend=backend, method=model.method)
    for index, plane in run_tasks(apply, tasks, total=total, execution=execution, workers=workers,
                                  backend=backend, method=model.method, operation="apply",
                                  plane_pixels=int(np.prod(input_view.shape[-2:])),
                                  progress=progress, cancelled=cancelled):
        output_view[np.unravel_index(index, navigation_shape)] = plane
    return output
