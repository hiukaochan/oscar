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

import copy
from hydra.core.config_store import ConfigStore

from  worldsim._ext.imaginaire.lazy_config import LazyCall as L
from  worldsim._ext.imaginaire.lazy_config import LazyDict
from worldsim._src.modules.selective_activation_checkpoint import SACConfig as SACConfig
from worldsim._src.networks.wan2pt1_i2v_concat import Wan2pt1I2VConcat
from worldsim._src.predict2.networks.minimal_lvg_v1_dit import MinimalV1LVGDiT
from worldsim._src.predict2.networks.minimal_v4_dit import SACConfig
# since we could only init from t2v for the i2v model, so the in_dim is smaller
WAN2PT1_I2V_1PT3B: LazyDict = L(Wan2pt1I2VConcat)(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=16,
    model_type="i2v",
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
    cp_comm_type="p2p",
    conv_patchify=True,
    additional_concat_ch=20+16,
    additional_embed_alpha=0.1,
    sac_config=L(SACConfig)(
        mode="block_wise"
    ),
)

# since we could init from original i2v model, so the in_dim is larger
WAN2PT1_I2V_14B: LazyDict = L(Wan2pt1I2VConcat)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=36,
    model_type="i2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
    cp_comm_type="p2p",
    conv_patchify=True,
    additional_concat_ch=16,
    additional_embed_alpha=0.1,
    sac_config=L(SACConfig)(
        mode="block_wise"
    ),
)

COSMOS_V1_7B_NET_MININET: LazyDict = L(MinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=4096,
    num_blocks=28,
    num_heads=32,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=True,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=2.0,
    sac_config=SACConfig(),
)
COSMOS_V1_2B_NET_MININET = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
COSMOS_V1_2B_NET_MININET.model_channels = 2048
COSMOS_V1_2B_NET_MININET.num_heads = 16
COSMOS_V1_2B_NET_MININET.num_blocks = 28
COSMOS_V1_2B_NET_MININET.extra_per_block_abs_pos_emb = False
COSMOS_V1_2B_NET_MININET.rope_t_extrapolation_ratio = 1.0

COSMOS_V1_14B_NET_MININET = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
COSMOS_V1_14B_NET_MININET.model_channels = 5120
COSMOS_V1_14B_NET_MININET.num_heads = 40
COSMOS_V1_14B_NET_MININET.num_blocks = 36
COSMOS_V1_14B_NET_MININET.extra_per_block_abs_pos_emb = False
COSMOS_V1_14B_NET_MININET.rope_t_extrapolation_ratio = 1.0

mini_net = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
mini_net.model_channels = 1024
mini_net.num_heads = 8
mini_net.num_blocks = 2
mini_net.rope_t_extrapolation_ratio = 1.0


COSMOS_V1_2B_NET_MININET_Concat: LazyDict = L(MinimalV1LVGDiT)(
    additional_concat_ch=16,
    additional_embed_alpha=0.1,
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
)


def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="wan2pt1_i2v_1pt3B", node=WAN2PT1_I2V_1PT3B)
    cs.store(group="net", package="model.config.net", name="wan2pt1_i2v_14B", node=WAN2PT1_I2V_14B)

    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B", node=COSMOS_V1_2B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_7B", node=COSMOS_V1_7B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_14B", node=COSMOS_V1_14B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B_concat", node=COSMOS_V1_2B_NET_MININET_Concat)