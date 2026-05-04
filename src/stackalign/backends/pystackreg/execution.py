from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from stackalign.constants import Method

from .utils import create_stackreg, identity_tmats

ExecutorMode = Literal["thread", "process"]

EXECUTOR_MODE: ExecutorMode = "process"
MAX_WORKERS: int | None = None


def create_executor(mode: ExecutorMode, max_workers: int | None = None) -> Executor:
    if mode == "thread":
        return ThreadPoolExecutor(max_workers=max_workers)
    if mode == "process":
        return ProcessPoolExecutor(max_workers=max_workers)
    raise ValueError(f"Unsupported executor mode: {mode!r}.")


def _register_frame_to_reference_task(frame_index: int, reference: NDArray[np.float32], moving: NDArray[np.float32], method: Method) -> tuple[int, NDArray[np.float64]]:
    sr = create_stackreg(method)
    tmat = sr.register(reference, moving)
    return frame_index, tmat


def _register_previous_pair_task(frame_index: int, previous: NDArray[np.float32], current: NDArray[np.float32], method: Method) -> tuple[int, NDArray[np.float64]]:
    sr = create_stackreg(method)
    tmat = sr.register(previous, current)
    return frame_index, tmat


def _apply_frame_tmat_task(frame_index: int, frame: NDArray[np.float32], tmat: NDArray[np.float64], method: Method) -> tuple[int, NDArray[np.float32]]:
    sr = create_stackreg(method)
    transformed = np.asarray(sr.transform(frame, tmat=tmat), dtype=np.float32)
    return frame_index, transformed


def _apply_channel_image_task(channel_index: int, image: NDArray[np.float32], tmat: NDArray[np.float64], method: Method) -> tuple[int, NDArray[np.float32]]:
    sr = create_stackreg(method)
    transformed = np.asarray(sr.transform(image, tmat=tmat), dtype=np.float32)
    return channel_index, transformed


def fit_frames_to_reference(fit_array_tyx: NDArray[np.float32], reference: NDArray[np.float32], method: Method) -> NDArray[np.float64]:
    tmats = identity_tmats(fit_array_tyx.shape[0])
    with create_executor(EXECUTOR_MODE, MAX_WORKERS) as executor:
        futures = [
            executor.submit(_register_frame_to_reference_task, frame_index, reference, fit_array_tyx[frame_index], method)
            for frame_index in range(fit_array_tyx.shape[0])
        ]
        for future in as_completed(futures):
            frame_index, tmat = future.result()
            tmats[frame_index] = tmat
    return tmats


def fit_previous_pairwise_tmats(fit_array_tyx: NDArray[np.float32], method: Method) -> NDArray[np.float64]:
    pairwise = identity_tmats(fit_array_tyx.shape[0])
    with create_executor(EXECUTOR_MODE, MAX_WORKERS) as executor:
        futures = [
            executor.submit(_register_previous_pair_task, frame_index, fit_array_tyx[frame_index - 1], fit_array_tyx[frame_index], method)
            for frame_index in range(1, fit_array_tyx.shape[0])
        ]
        for future in as_completed(futures):
            frame_index, tmat = future.result()
            pairwise[frame_index] = tmat
    return pairwise


def apply_tyx_substack(substack_tyx: NDArray[np.float32], tmats: NDArray[np.float64], method: Method) -> NDArray[np.float32]:
    if substack_tyx.shape[0] != tmats.shape[0]:
        raise ValueError(f"Time model length must match array T length. Got {tmats.shape[0]} transforms for T={substack_tyx.shape[0]}.")

    transformed_substack = np.empty_like(substack_tyx, dtype=np.float32)

    if substack_tyx.shape[0] <= 2 or MAX_WORKERS == 1:
        sr = create_stackreg(method)
        for frame_index in range(substack_tyx.shape[0]):
            transformed_substack[frame_index] = sr.transform(substack_tyx[frame_index], tmat=tmats[frame_index])
        return transformed_substack

    with create_executor(EXECUTOR_MODE, MAX_WORKERS) as executor:
        futures = [
            executor.submit(_apply_frame_tmat_task, frame_index, substack_tyx[frame_index], tmats[frame_index], method)
            for frame_index in range(substack_tyx.shape[0])
        ]
        for future in as_completed(futures):
            frame_index, transformed_frame = future.result()
            transformed_substack[frame_index] = transformed_frame

    return transformed_substack


def apply_cyx_substack(substack_cyx: NDArray[np.float32], tmats: NDArray[np.float64], method: Method, reference_channel: int) -> NDArray[np.float32]:
    if substack_cyx.shape[0] != tmats.shape[0]:
        raise ValueError(f"Channel model length must match array C length. Got {tmats.shape[0]} transforms for C={substack_cyx.shape[0]}.")
    if not (0 <= reference_channel < substack_cyx.shape[0]):
        raise ValueError(f"reference_channel={reference_channel} is out of range for C={substack_cyx.shape[0]}.")

    transformed_substack = np.empty_like(substack_cyx, dtype=np.float32)
    transformed_substack[reference_channel] = substack_cyx[reference_channel]

    if substack_cyx.shape[0] <= 2 or MAX_WORKERS == 1:
        sr = create_stackreg(method)
        for channel_index in range(substack_cyx.shape[0]):
            if channel_index == reference_channel:
                continue
            transformed_substack[channel_index] = sr.transform(substack_cyx[channel_index], tmat=tmats[channel_index])
        return transformed_substack

    with create_executor(EXECUTOR_MODE, MAX_WORKERS) as executor:
        futures = [
            executor.submit(_apply_channel_image_task, channel_index, substack_cyx[channel_index], tmats[channel_index], method)
            for channel_index in range(substack_cyx.shape[0])
            if channel_index != reference_channel
        ]
        for future in as_completed(futures):
            channel_index, transformed_channel = future.result()
            transformed_substack[channel_index] = transformed_channel

    return transformed_substack