# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from enum import Enum


class ErrorCode(Enum):
    INIT_VALUE = -1
    SUCCESS = 0
    ABNORMAL_INTERRUPTION = 1
    OUT_OF_MAX_STEP = 2

    UNKNOWN_ERROR = 500
