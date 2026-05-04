from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from stackalign.axes import canonical_axes, move_to_axes, normalize_axes, require_axes_member

@dataclass(slots=True)
class FitPreparation:
    """
    Prepared arrays and metadata for fitting workflows.
    """
    original_array: NDArray[np.generic]
    original_axes: str
    fit_array: NDArray[np.float32] | None
    fit_axes: str | None

    @classmethod
    def for_time(cls, array: NDArray[np.generic], axes: str, fit_channel: int | None = None) -> FitPreparation:
        source = np.asarray(array)
        source_axes = normalize_axes(source, axes)
        require_axes_member(source_axes, "T", "fit_time")

        canonical_source_axes = canonical_axes(source_axes)
        canonical_source = move_to_axes(source, source_axes, canonical_source_axes)
        fit_array, fit_axes = cls._build_time_fit_view(source_array=canonical_source, source_axes=canonical_source_axes, fit_channel=fit_channel)

        return cls(
            original_array=source,
            original_axes=source_axes,
            fit_array=fit_array,
            fit_axes=fit_axes,)

    @staticmethod
    def _build_time_fit_view(source_array: NDArray[np.generic], source_axes: str, fit_channel: int | None) -> tuple[NDArray[np.float32], str]:
        work = source_array
        work_axes = source_axes

        if "C" in work_axes:
            if fit_channel is None:
                raise ValueError("fit_time with a C axis requires fit_channel to select the fitting channel.")
            
            c_index = work_axes.index("C")
            if not (0 <= fit_channel < work.shape[c_index]):
                raise ValueError(f"fit_channel={fit_channel} is out of range for C={work.shape[c_index]}.")
            
            work = np.take(work, indices=fit_channel, axis=c_index)
            work_axes = work_axes.replace("C", "")
        elif fit_channel is not None:
            raise ValueError("fit_channel was provided but axes does not include C.")

        if "Z" in work_axes:
            z_index = work_axes.index("Z")
            work = np.asarray(work, dtype=np.float32).max(axis=z_index)
            work_axes = work_axes.replace("Z", "")

        if sorted(work_axes) != sorted("TYX"):
            raise ValueError(f"fit_time expects axes containing at least T, Y, and X after channel/Z handling. Got remaining axes='{work_axes}'.")

        fit_array = move_to_axes(np.asarray(work), work_axes, "TYX")
        if fit_array.shape[0] == 0:
            raise ValueError("fit_time requires at least one frame. Got T=0.")
        return np.asarray(fit_array, dtype=np.float32), "TYX"

    @classmethod
    def for_channel(cls, array: NDArray[np.generic], axes: str, reference_channel: int | None, reference_frame: int = 0) -> FitPreparation:
        source = np.asarray(array)
        source_axes = normalize_axes(source, axes)
        require_axes_member(source_axes, "C", "fit_channel")

        canonical_source_axes = canonical_axes(source_axes)
        canonical_source = move_to_axes(source, source_axes, canonical_source_axes)
        fit_array, fit_axes = cls._build_channel_fit_view(
            source_array=canonical_source,
            source_axes=canonical_source_axes,
            reference_channel=reference_channel,
            reference_frame=reference_frame,)

        return cls(
            original_array=source,
            original_axes=source_axes,
            fit_array=fit_array,
            fit_axes=fit_axes,)

    @staticmethod
    def _build_channel_fit_view(source_array: NDArray[np.generic], source_axes: str, reference_channel: int | None, reference_frame: int = 0) -> tuple[NDArray[np.float32], str]:
        work = source_array
        work_axes = source_axes

        c_index = work_axes.index("C")
        if reference_channel is None:
            raise ValueError("fit_channel requires reference_channel.")
        if not (0 <= reference_channel < work.shape[c_index]):
            raise ValueError(f"reference_channel={reference_channel} is out of range for C={work.shape[c_index]}.")

        if "T" in work_axes:
            t_index = work_axes.index("T")
            if not (0 <= reference_frame < work.shape[t_index]):
                raise ValueError(f"reference_frame={reference_frame} is out of range for T={work.shape[t_index]}.")
            work = np.take(work, indices=reference_frame, axis=t_index)
            work_axes = work_axes.replace("T", "")
        elif reference_frame != 0:
            raise ValueError("reference_frame requires a T axis unless it is 0.")

        if "Z" in work_axes:
            z_index = work_axes.index("Z")
            work = np.asarray(work, dtype=np.float32).max(axis=z_index)
            work_axes = work_axes.replace("Z", "")

        if sorted(work_axes) != sorted("CYX"):
            raise ValueError(f"fit_channel expects axes containing at least C, Y, and X after frame/Z handling. Got remaining axes='{work_axes}'.")

        fit_array = move_to_axes(np.asarray(work), work_axes, "CYX")
        return np.asarray(fit_array, dtype=np.float32), "CYX"


@dataclass(slots=True)
class ApplyPreparation:
    """
    Prepared arrays and metadata for apply workflows.
    """
    original_array: NDArray[np.generic]
    original_axes: str
    original_dtype: np.dtype[np.generic]
    original_shape: tuple[int, ...]
    apply_array: NDArray[np.generic]
    apply_axes: str

    @classmethod
    def for_time(cls, array: NDArray[np.generic], axes: str) -> ApplyPreparation:
        source = np.asarray(array)
        source_axes = normalize_axes(source, axes)
        require_axes_member(source_axes, "T", "apply_time")

        apply_axes = canonical_axes(source_axes)
        apply_array = move_to_axes(source, source_axes, apply_axes)

        return cls(
            original_array=source,
            original_axes=source_axes,
            original_dtype=source.dtype,
            original_shape=source.shape,
            apply_array=apply_array,
            apply_axes=apply_axes,)

    @classmethod
    def for_channel(cls, array: NDArray[np.generic], axes: str) -> ApplyPreparation:
        source = np.asarray(array)
        source_axes = normalize_axes(source, axes)
        require_axes_member(source_axes, "C", "apply_channel")

        apply_axes = canonical_axes(source_axes)
        apply_array = move_to_axes(source, source_axes, apply_axes)

        return cls(
            original_array=source,
            original_axes=source_axes,
            original_dtype=source.dtype,
            original_shape=source.shape,
            apply_array=apply_array,
            apply_axes=apply_axes,)

    def iter_apply_substacks(self, required_axes: str) -> Iterator[tuple[tuple[slice | int, ...], NDArray[np.float32]]]:
        """
        Yield pairs of (slicer, substack) for iterating through substacks with the given required axes.
        
        Iterates through all non-required axes, yielding the corresponding substack data.
        For example, required_axes="TYX" iterates through all C/Z combinations.
        
        Args:
            required_axes: Axes that must be present and preserved (e.g., "TYX" or "CYX").
        """
        if not all(axis in self.apply_axes for axis in required_axes):
            raise ValueError(f"apply_axes must include {required_axes}. Got '{self.apply_axes}'.")

        extra_axes = "".join(axis for axis in self.apply_axes if axis not in required_axes)
        if not extra_axes:
            slicer = tuple(slice(None) for _ in self.apply_axes)
            yield slicer, np.asarray(self.apply_array, dtype=np.float32)
            return

        extra_shape = tuple(self.apply_array.shape[self.apply_axes.index(axis)] for axis in extra_axes)
        for extra_index in np.ndindex(extra_shape):
            slicer_list: list[slice | int] = [slice(None)] * len(self.apply_axes)
            for axis, value in zip(extra_axes, extra_index):
                slicer_list[self.apply_axes.index(axis)] = value
            slicer = tuple(slicer_list)
            yield slicer, np.asarray(self.apply_array[slicer], dtype=np.float32)

    def iter_apply_tyx_substacks(self) -> Iterator[tuple[tuple[slice | int, ...], NDArray[np.float32]]]:
        """Iterate through TYX substacks. Convenience wrapper."""
        yield from self.iter_apply_substacks("TYX")

    def iter_apply_cyx_substacks(self) -> Iterator[tuple[tuple[slice | int, ...], NDArray[np.float32]]]:
        """Iterate through CYX substacks. Convenience wrapper."""
        yield from self.iter_apply_substacks("CYX")

    def restore_apply_output(self, transformed_apply: NDArray[np.floating]) -> NDArray[np.generic]:
        """
        Restore the transformed apply array to the original axes, dtype, and shape of the input apply array. This includes moving axes back to the original order and converting dtype back to the original dtype with appropriate clipping for integer types.
        """
        if transformed_apply.shape != self.apply_array.shape:
            raise ValueError(f"Transformed apply array shape mismatch. Expected {self.apply_array.shape}, got {transformed_apply.shape}.")

        output = move_to_axes(transformed_apply, self.apply_axes, self.original_axes)
        return self._restore_dtype(output, self.original_dtype)

    @staticmethod
    def _restore_dtype(array: NDArray[np.floating], target_dtype: np.dtype[np.generic]) -> NDArray[np.generic]:
        if array.dtype == target_dtype:
            return array

        if np.issubdtype(target_dtype, np.integer):
            info = np.iinfo(np.dtype(target_dtype).name)
            clipped = np.clip(np.rint(array), info.min, info.max)
            return clipped.astype(target_dtype, copy=False)

        return array.astype(target_dtype, copy=False)
