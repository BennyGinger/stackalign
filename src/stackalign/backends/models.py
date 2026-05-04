from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from stackalign.constants import Method


@dataclass(slots=True)
class TransformModel:
    """Tiny internal fitted model for time-wise or channel-wise transforms.
    
    Attributes
    ----------
    mode:
    	Whether this is a time-wise or channel-wise model. 
    method:
        Registration method: translation, rigid_body, affine.
    transform:
        Explicit transform matrices with shape (N, 3, 3). N is T for time-wise
        models and C for channel-wise models.
    reference_channel:
        For channel-wise models, the index of the reference channel. None for time-wise models.
    """

    mode: Literal["time", "channel"]
    method: Method
    transform: NDArray[np.float64]
    reference_channel: int | None = None