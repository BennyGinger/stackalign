from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from stackalign.backends.transforms import identity_tmats
from stackalign.backends.protocol import ChannelApplyFn, FrameApplyFn, FrameFitFn, PairFitFn

ExecutorMode = Literal["thread", "process"]

EXECUTOR_MODE: ExecutorMode = "process" 
MAX_WORKERS: int | None = None


def create_executor(mode: ExecutorMode, max_workers: int | None = None) -> Executor:
    if mode == "thread":
        return ThreadPoolExecutor(max_workers=max_workers)
    if mode == "process":
        return ProcessPoolExecutor(max_workers=max_workers)
    raise ValueError(f"Unsupported executor mode: {mode!r}.")


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
    with create_executor(EXECUTOR_MODE, MAX_WORKERS) as executor:
        futures = [
            executor.submit(frame_apply_fn, frame_index, substack_tyx[frame_index], tmats[frame_index])
            for frame_index in range(substack_tyx.shape[0])
            ]
        
        for future in as_completed(futures):
            frame_index, transformed_frame = future.result()
            transformed[frame_index] = transformed_frame
    return transformed


def apply_cyx_substack(substack_cyx: NDArray[np.float32], tmats: NDArray[np.float64], reference_channel: int, channel_apply_fn: ChannelApplyFn) -> NDArray[np.float32]:
    """
    Apply per-channel transforms to a CYX substack using *channel_apply_fn*.

    The reference channel is copied unchanged. Non-reference channels are
    processed serially in-process using the signature::

        channel_apply_fn(channel_index, image, tmat) -> (channel_index, transformed)

    Use :func:`functools.partial` to bind any extra backend-specific arguments.
    """
    if substack_cyx.shape[0] != tmats.shape[0]:
        raise ValueError(f"Channel model length must match array C length. Got {tmats.shape[0]} transforms for C={substack_cyx.shape[0]}.")
    
    if not (0 <= reference_channel < substack_cyx.shape[0]):
        raise ValueError(f"reference_channel={reference_channel} is out of range for C={substack_cyx.shape[0]}.")
    
    transformed = np.empty_like(substack_cyx, dtype=np.float32)
    transformed[reference_channel] = substack_cyx[reference_channel]
    for channel_index in range(substack_cyx.shape[0]):
        if channel_index == reference_channel:
            continue
        _, transformed_channel = channel_apply_fn(channel_index, substack_cyx[channel_index], tmats[channel_index])
        transformed[channel_index] = transformed_channel
    return transformed


def fit_frames_to_reference(fit_array_tyx: NDArray[np.float32], reference: NDArray[np.float32], frame_task_fn: FrameFitFn) -> NDArray[np.float64]:
    """
    Run frame-to-reference registration in parallel using *frame_task_fn*.

    *frame_task_fn* must be a top-level, picklable callable with the signature::

        frame_task_fn(frame_index, reference, moving) -> (frame_index, tmat)

    Use :func:`functools.partial` to bind any extra backend-specific arguments.
    """
    tmats = identity_tmats(fit_array_tyx.shape[0])
    with create_executor(EXECUTOR_MODE, MAX_WORKERS) as executor:
        futures = [
            executor.submit(frame_task_fn, frame_index, reference, fit_array_tyx[frame_index])
            for frame_index in range(fit_array_tyx.shape[0])
        ]
        
        for future in as_completed(futures):
            frame_index, tmat = future.result()
            tmats[frame_index] = tmat
    return tmats


def fit_previous_pairwise(fit_array_tyx: NDArray[np.float32], pair_task_fn: PairFitFn) -> NDArray[np.float64]:
    """
    Run pairwise (t-1 -> t) registration in parallel using *pair_task_fn*.

    *pair_task_fn* must be a top-level, picklable callable with the signature::

        pair_task_fn(frame_index, previous, current) -> (frame_index, tmat)

    Use :func:`functools.partial` to bind any extra backend-specific arguments.
    """
    pairwise = identity_tmats(fit_array_tyx.shape[0])
    with create_executor(EXECUTOR_MODE, MAX_WORKERS) as executor:
        futures = [
            executor.submit(pair_task_fn, frame_index, fit_array_tyx[frame_index - 1], fit_array_tyx[frame_index])
            for frame_index in range(1, fit_array_tyx.shape[0])
        ]
        
        for future in as_completed(futures):
            frame_index, tmat = future.result()
            pairwise[frame_index] = tmat
    return pairwise
