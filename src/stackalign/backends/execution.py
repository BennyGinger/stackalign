from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, ProcessPoolExecutor, ThreadPoolExecutor, wait
from multiprocessing import get_context
import os
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from stackalign.backends.transforms import identity_tmats
from stackalign.backends.protocol import ChannelApplyFn, FrameApplyFn, FrameFitFn, PairFitFn
from stackalign.constants import Method

ExecutionMode = Literal["auto", "serial", "thread", "process"]
ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]

EXECUTOR_MODE: ExecutionMode = "auto"
MAX_WORKERS: int | None = None


def validate_execution(mode: str, workers: int | None) -> None:
    """Reject unsupported execution modes and invalid worker counts."""
    if mode not in ("auto", "serial", "thread", "process"):
        raise ValueError(f"Unsupported execution mode {mode!r}; use auto, serial, thread, or process.")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int) or workers < 1):
        raise ValueError("workers must be a positive integer or None.")


def available_cpus() -> int:
    """Count CPUs available to this process, respecting affinity where supported."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def available_memory() -> int | None:
    """Read currently available physical memory where the OS exposes sysconf."""
    try:
        value = int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        return value if value > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def select_execution(mode: ExecutionMode,
                     *, backend: str | None, method: Method, operation: Literal["fit", "apply"],
                     plane_pixels: int, total: int, workers: int | None = None) -> ExecutionMode:
    """Resolve auto from rough benchmark costs, job count, CPUs and process memory."""
    validate_execution(mode, workers)
    if mode != "auto":
        return mode
    worker_count = min(total, workers or 4, available_cpus())
    if worker_count < 2 or plane_pixels < 128**2:
        return "serial"
    # The benchmark had three fit jobs and sixteen apply jobs. Total pixels
    # approximate work while the worker factor avoids assuming four-way gains
    # when only two CPUs/jobs are available. These are deliberately rough rules.
    capacity = min(worker_count, 4) / (3 if operation == "fit" else 4)
    work = plane_pixels * total * capacity
    if backend == "pystackreg":
        threshold = (3 * (512 if method == "affine" else 1024)**2
                     if operation == "fit" else 16 * 512**2)
        if work < threshold:
            return "serial"
        memory = available_memory()
        # Allow for spawned interpreters, queued planes and native workspaces.
        # This guard is an estimate, not a promise of bounded native RAM use.
        working_bytes = worker_count * (128 * 1024**2 + 64 * plane_pixels)
        if memory is not None and working_bytes > memory // 2:
            return "serial"
        return "process"
    if backend in ("cv2", "scikit"):
        threshold = (3 * 256**2 if operation == "fit"
                     else 16 * (128 if backend == "scikit" else 256)**2)
        return "thread" if work >= threshold else "serial"
    return "serial"


def check_cancelled(cancelled: CancelCallback | None) -> None:
    """Raise InterruptedError when the caller requests cancellation."""
    if cancelled is not None and cancelled():
        raise InterruptedError("Registration cancelled.")


def create_executor(mode: ExecutionMode, max_workers: int | None = None) -> Executor:
    """Create a bounded thread pool or a process pool using spawn."""
    validate_execution(mode, max_workers)
    max_workers = max_workers or min(4, available_cpus())
    if mode == "thread":
        return ThreadPoolExecutor(max_workers=max_workers)
    if mode == "process":
        # Safe when called from a GUI worker thread, including on Windows.
        return ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn"))
    raise ValueError(f"Unsupported executor mode: {mode!r}.")


def run_tasks(task_fn: Callable[..., tuple[int, NDArray]],
              tasks: Iterable[Sequence[Any]],
              *,
              total: int,
              execution: ExecutionMode | None = None,
              workers: int | None = None,
              backend: str | None = None,
              method: Method = "translation",
              operation: Literal["fit", "apply"] = "fit",
              plane_pixels: int = 0,
              progress: ProgressCallback | None = None,
              cancelled: CancelCallback | None = None
              ) -> Iterator[tuple[int, NDArray]]:
    """
    Run a bounded queue; callbacks stay in the calling thread, never in workers.
    """
    execution = execution if execution is not None else EXECUTOR_MODE
    workers = workers if workers is not None else MAX_WORKERS
    validate_execution(execution, workers)
    automatic = execution == "auto"
    execution = select_execution(execution, backend=backend, method=method, operation=operation,
                                 plane_pixels=plane_pixels, total=total, workers=workers)
    check_cancelled(cancelled)
    if progress is not None:
        progress(0, total)
    check_cancelled(cancelled)
    if total == 0:
        return
    if execution == "serial":
        for done, arguments in enumerate(tasks, 1):
            check_cancelled(cancelled)
            result = task_fn(*arguments)
            check_cancelled(cancelled)
            yield result
            if progress is not None:
                progress(done, total)
        check_cancelled(cancelled)
        return

    worker_count = min(total, workers or min(4, available_cpus()))
    if automatic:
        worker_count = min(worker_count, available_cpus())
    iterator = iter(tasks)
    executor = create_executor(execution, worker_count)
    pending = set()
    done_count = 0
    try:
        def submit_next() -> bool:
            """Submit the next task unless the iterator is exhausted or cancelled."""
            check_cancelled(cancelled)
            arguments = next(iterator, None)
            if arguments is None:
                return False
            pending.add(executor.submit(task_fn, *arguments))
            return True

        for _ in range(2 * worker_count):
            if not submit_next():
                break
        while pending:
            check_cancelled(cancelled)
            completed, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.remove(future)
                check_cancelled(cancelled)
                yield future.result()
                done_count += 1
                if progress is not None:
                    progress(done_count, total)
                submit_next()
        check_cancelled(cancelled)
    finally:
        for future in pending:
            future.cancel()
        # Native calls already running finish before input/output storage is released.
        executor.shutdown(wait=True, cancel_futures=True)


def apply_tyx_substack(substack_tyx: NDArray[np.float32], tmats: NDArray[np.float64], frame_apply_fn: FrameApplyFn) -> NDArray[np.float32]:
    """
    Apply per-frame transforms to a TYX substack using *frame_apply_fn*.

    *frame_apply_fn* must be a top-level, picklable callable with the signature::

        frame_apply_fn(frame_index, frame, tmat) -> (frame_index, transformed_frame)

    Use :func:`functools.partial` to bind any extra backend-specific arguments.
    """
    if substack_tyx.shape[0] != tmats.shape[0]:
        raise ValueError(f"Time model length must match array T length. Got {tmats.shape[0]} transforms for T={substack_tyx.shape[0]}.")

    transformed = np.empty_like(substack_tyx, dtype=np.float32)
    tasks = ((index, substack_tyx[index], tmats[index]) for index in range(len(substack_tyx)))
    for index, plane in run_tasks(frame_apply_fn, tasks, total=len(substack_tyx)):
        transformed[index] = plane
    return transformed


def apply_cyx_substack(substack_cyx: NDArray[np.float32], tmats: NDArray[np.float64], reference_channel: int, channel_apply_fn: ChannelApplyFn) -> NDArray[np.float32]:
    """
    Apply per-channel transforms to a CYX substack using *channel_apply_fn*.

    The reference channel is copied unchanged. Other channels use the selected
    execution mode with the signature::

        channel_apply_fn(channel_index, image, tmat) -> (channel_index, transformed)

    Use :func:`functools.partial` to bind any extra backend-specific arguments.
    """
    if substack_cyx.shape[0] != tmats.shape[0]:
        raise ValueError(f"Channel model length must match array C length. Got {tmats.shape[0]} transforms for C={substack_cyx.shape[0]}.")

    if not (0 <= reference_channel < substack_cyx.shape[0]):
        raise ValueError(f"reference_channel={reference_channel} is out of range for C={substack_cyx.shape[0]}.")

    transformed = np.empty_like(substack_cyx, dtype=np.float32)
    transformed[reference_channel] = substack_cyx[reference_channel]
    tasks = ((index, substack_cyx[index], tmats[index]) for index in range(len(substack_cyx))
             if index != reference_channel)
    for index, plane in run_tasks(channel_apply_fn, tasks, total=len(substack_cyx) - 1):
        transformed[index] = plane
    return transformed


def fit_frames_to_reference(fit_array_tyx: NDArray[np.float32], reference: NDArray[np.float32], frame_task_fn: FrameFitFn) -> NDArray[np.float64]:
    """
    Run frame-to-reference registration in parallel using *frame_task_fn*.

    *frame_task_fn* must be a top-level, picklable callable with the signature::

        frame_task_fn(frame_index, reference, moving) -> (frame_index, tmat)

    Use :func:`functools.partial` to bind any extra backend-specific arguments.
    """
    tmats = identity_tmats(fit_array_tyx.shape[0])
    tasks = ((index, reference, fit_array_tyx[index]) for index in range(len(fit_array_tyx)))
    for index, matrix in run_tasks(frame_task_fn, tasks, total=len(fit_array_tyx)):
        tmats[index] = matrix
    return tmats


def fit_previous_pairwise(fit_array_tyx: NDArray[np.float32], pair_task_fn: PairFitFn) -> NDArray[np.float64]:
    """
    Run pairwise (t-1 -> t) registration in parallel using *pair_task_fn*.

    *pair_task_fn* must be a top-level, picklable callable with the signature::

        pair_task_fn(frame_index, previous, current) -> (frame_index, tmat)

    Use :func:`functools.partial` to bind any extra backend-specific arguments.
    """
    pairwise = identity_tmats(fit_array_tyx.shape[0])
    tasks = ((index, fit_array_tyx[index - 1], fit_array_tyx[index])
             for index in range(1, len(fit_array_tyx)))
    for index, matrix in run_tasks(pair_task_fn, tasks, total=len(fit_array_tyx) - 1):
        pairwise[index] = matrix
    return pairwise
