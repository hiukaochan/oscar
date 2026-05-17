# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from dataclasses import dataclass
from typing import Dict, Optional
import copy
import torch
from einops import rearrange
from hydra.core.config_store import ConfigStore
from torch.distributed import get_process_group_ranks

from worldsim._src.modules.conditioner import (
    BooleanFlag,
    GeneralConditioner,
    ReMapkey,
    T2VCondition,
    TextAttr,
    TextAttrEmptyStringDrop,
)
from worldsim._src.utils.context_parallel import broadcast_split_tensor, find_split

from worldsim._ext.imaginaire.lazy_config import LazyCall as L
from worldsim._ext.imaginaire.lazy_config import LazyDict
from worldsim._src.conditioner.cosmos2_v2v_conditioner import Video2WorldConditioner, Video2WorldConditionerV2, ControlVideo2WorldConditioner
from worldsim._src.predict2.models.video2world_rectified_flow import COSMOS_CONTROL_KEY


_SHARED_CONFIG = dict(
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
        use_empty_string=False,
    ),
    use_video_condition=L(BooleanFlag)(
        input_key="fps",
        output_key="use_video_condition",
        dropout_rate=0.2,
    ),
)

_CONTROL_CONFIG = copy.deepcopy(_SHARED_CONFIG)
_CONTROL_CONFIG.update(
    dict(
        control_output=L(ReMapkey)(
            input_key=COSMOS_CONTROL_KEY,
            output_key=COSMOS_CONTROL_KEY,
            dropout_rate=0.0,
            dtype=None,
        ),
    )
)


VideoPredictionConditioner: LazyDict = L(Video2WorldConditioner)(
    **_SHARED_CONFIG,
)

VideoPredictionConditionerV2: LazyDict = L(Video2WorldConditionerV2)(
    **_SHARED_CONFIG,
)

ControlVideoPredictionConditioner: LazyDict = L(ControlVideo2WorldConditioner)(
    **_CONTROL_CONFIG,
)



def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner",
        node=VideoPredictionConditioner,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner_v2",
        node=VideoPredictionConditionerV2,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="control_video_prediction_conditioner",
        node=ControlVideoPredictionConditioner,
    )
