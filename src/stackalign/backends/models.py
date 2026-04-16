from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pystackreg import StackReg


@dataclass(slots=True)
class TransformModel:
    """Tiny internal fitted model for time-wise or channel-wise transforms."""

    mode: Literal["time", "channel"]
    transform: StackReg | list[StackReg | None]
    reference_channel: int | None = None