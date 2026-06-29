# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from typing import NamedTuple
from jaxtyping import Float
import numpy as np


class BBox(NamedTuple):
    # A n-dim box.
    center: Float[np.ndarray, "n"]
    min: Float[np.ndarray, "n"]
    max: Float[np.ndarray, "n"]
    sizes: Float[np.ndarray, "n"]
    size: float
