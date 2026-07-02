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


from hydra.core.config_store import ConfigStore

from  worldsim._ext.imaginaire.lazy_config import LazyCall as L
from worldsim._src.models.wan_i2v_control_model import I2VWan2pt1ControlModel
from worldsim._src.models.wan_t2v_model import T2VModelConfig
from worldsim._src.predict2.models.video2world_rectified_flow import (
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
    CosmosI2VControlModel,
    CosmosJointRGBSkelModel,
)

ddp_wan2pt1_config = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(I2VWan2pt1ControlModel)(
        config=T2VModelConfig(
            state_t=20,
        ),
        _recursive_=False,
    ),
)


fsdp_wan2pt1_config = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(I2VWan2pt1ControlModel)(
        config=T2VModelConfig(
            fsdp_shard_size=8,
            state_t=20,
        ),
        _recursive_=False,
    ),
)

FSDP_RECTIFIED_FLOW_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Video2WorldModelRectifiedFlow)(
        config=Video2WorldModelRectifiedFlowConfig(
            fsdp_shard_size=8,
            state_t=24,
        ),
        _recursive_=False,
    ),
)

FSDP_RECTIFIED_FLOW_CONFIG_Control = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CosmosI2VControlModel)(
        config=Video2WorldModelRectifiedFlowConfig(
            fsdp_shard_size=8,
            state_t=24,
        ),
        _recursive_=False,
    ),
)



FSDP_RECTIFIED_FLOW_CONFIG_Joint = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(CosmosJointRGBSkelModel)(
        config=Video2WorldModelRectifiedFlowConfig(
            fsdp_shard_size=8,
            state_t=24,
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp_iv2_control", node=ddp_wan2pt1_config)
    cs.store(group="model", package="_global_", name="fsdp_i2v_control", node=fsdp_wan2pt1_config)
    cs.store(group="model", package="_global_", name="fsdp_cosmos_i2v_rectified_flow", node=FSDP_RECTIFIED_FLOW_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_cosmos_i2v_rectified_flow_control", node=FSDP_RECTIFIED_FLOW_CONFIG_Control)
    cs.store(group="model", package="_global_", name="fsdp_cosmos_i2v_rectified_flow_joint", node=FSDP_RECTIFIED_FLOW_CONFIG_Joint)
