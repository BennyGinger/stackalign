from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from stackalign.constants import AXIS_LABELS, CANONICAL_AXIS_ORDER, DType


def normalize_axes(array: NDArray[np.generic], axes: str) -> str:
    """
    Validate an explicit axes string and return it uppercased.

    Supported axis labels are T, C, Z, Y, and X. The axes string may be in any
    order, but must include X and Y.
    """
    if axes is None:
        raise ValueError("axes is required and cannot be None.")
    normalized = axes.upper()
    if len(normalized) != array.ndim:
        raise ValueError(f"axes='{normalized}' has length {len(normalized)} but array has ndim={array.ndim}.")
    
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"axes must not contain duplicates. Got '{normalized}'.")
    invalid = [axis for axis in normalized if axis not in AXIS_LABELS]
    if invalid:
        raise ValueError(f"Unsupported axis label(s) {invalid} in axes='{axes}'. Supported labels: {sorted(AXIS_LABELS)}")
    
    require_axes_member(normalized, "X", "normalize_axes")
    require_axes_member(normalized, "Y", "normalize_axes")
    return normalized


def require_axes_member(axes: str, required: str, context: str) -> None:
    """
    Raise a clear error when an expected axis is missing.
    """
    if required not in axes:
        raise ValueError(f"{context} requires axis '{required}', but axes='{axes}'.")


def canonical_axes(axes: str) -> str:
    """
    Return canonical axis order (subset of TCZYX) for a validated axes string.
    """
    return "".join(axis for axis in CANONICAL_AXIS_ORDER if axis in axes)


def move_to_axes(array: NDArray[DType], source_axes: str, target_axes: str) -> NDArray[DType]:
    """
    Return a view of array reordered from source_axes to target_axes.
    """
    if sorted(source_axes) != sorted(target_axes):
        raise ValueError(f"source_axes and target_axes must contain the same axis labels. Got source='{source_axes}', target='{target_axes}'.")
    
    if source_axes == target_axes:
        return array
    
    mapping = [source_axes.index(axis) for axis in target_axes]
    return np.moveaxis(array, mapping, range(len(mapping)))
