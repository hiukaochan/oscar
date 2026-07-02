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

"""Joint RGB + skeleton DiT.

Unlike ``MinimalV1LVGDiT`` (which fuses a clean, never-noised skeleton latent into
the RGB stream via weighted-sum at the embedding level), this network treats
skeleton as a second, independently-noised denoising target. RGB and skeleton are
patch-embedded independently, tagged with learned modality embeddings, and
concatenated along the temporal axis so they share one self-attention sequence
through the (unmodified) transformer backbone -- giving full joint bidirectional
attention between the two streams for free. After the last block the sequence is
split back into RGB/skeleton halves and projected out through two independent
``FinalLayer`` heads (no shared weights).
"""

from typing import List, Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torchvision import transforms

from worldsim._src.modules.conditioner import DataType
from worldsim._src.predict2.networks.minimal_v4_dit import (
    CheckpointMode,
    FinalLayer,
    MiniTrainDIT,
    PatchEmbed,
    SACConfig,
)


class MinimalV1JointDiT(MiniTrainDIT):
    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # condition-mask channel; added to BOTH streams in forward()
        self.timestep_scale = timestep_scale
        self._sac_config = kwargs.get("sac_config", SACConfig())
        super().__init__(*args, **kwargs)

        # MiniTrainDIT.__init__ has no build_final_layer() hook -- it unconditionally builds a
        # single `self.final_layer` (already initialized and, if sac_config enables it, already
        # activation-checkpoint-wrapped, by the time super().__init__() returns). Reuse that
        # instance as head_rgb, and build an independent, identically-configured head_skel.
        self.head_rgb = self.final_layer
        del self.final_layer
        self.head_skel = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
            use_wan_fp32_strategy=self.use_wan_fp32_strategy,
        )
        if self._sac_config.mode != CheckpointMode.NONE:
            self.head_skel = ptd_checkpoint_wrapper(
                self.head_skel, context_fn=self._sac_config.get_context_fn(), preserve_rng_state=False
            )

    def build_patch_embed(self):
        concat_padding_mask, in_channels, patch_spatial, patch_temporal, model_channels = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )
        self.skel_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )
        self.modality_embed_rgb = nn.Parameter(torch.zeros(1, 1, 1, 1, model_channels))
        self.modality_embed_skel = nn.Parameter(torch.zeros(1, 1, 1, 1, model_channels))

    def init_weights(self):
        super().init_weights()
        self.skel_embedder.init_weights()
        nn.init.zeros_(self.modality_embed_rgb)
        nn.init.zeros_(self.modality_embed_skel)

    def prepare_joint_embedded_sequence(
        self,
        x_rgb_B_C_T_H_W: torch.Tensor,
        x_skel_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Patch-embed RGB/skeleton independently, tag with modality embeddings,
        concatenate along T, and build a rope table via TILING (not range
        extension) so that RGB-frame-i and skeleton-frame-i share an identical
        rotary position code.

        Returns (x_B_2T_H_W_D, rope_emb_2THW_1_1_D, T) where T is the per-stream
        (pre-concat) latent frame count, needed by the caller to split the
        sequence back into RGB/skeleton halves after the transformer blocks.
        """
        assert not self.extra_per_block_abs_pos_emb, (
            "MinimalV1JointDiT does not support extra_per_block_abs_pos_emb: tiling "
            "LearnablePosEmbAxis under temporal-axis doubling is unimplemented."
        )
        assert "rope" in self.pos_emb_cls.lower(), "MinimalV1JointDiT only supports rope3d positional embeddings."

        if self.concat_padding_mask:
            resized_padding_mask = transforms.functional.resize(
                padding_mask, list(x_rgb_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            pad_channel_rgb = resized_padding_mask.unsqueeze(1).repeat(1, 1, x_rgb_B_C_T_H_W.shape[2], 1, 1)
            x_rgb_B_C_T_H_W = torch.cat([x_rgb_B_C_T_H_W, pad_channel_rgb.type_as(x_rgb_B_C_T_H_W)], dim=1)
            pad_channel_skel = resized_padding_mask.unsqueeze(1).repeat(1, 1, x_skel_B_C_T_H_W.shape[2], 1, 1)
            x_skel_B_C_T_H_W = torch.cat([x_skel_B_C_T_H_W, pad_channel_skel.type_as(x_skel_B_C_T_H_W)], dim=1)

        x_rgb_B_T_H_W_D = self.x_embedder(x_rgb_B_C_T_H_W)
        x_skel_B_T_H_W_D = self.skel_embedder(x_skel_B_C_T_H_W)

        x_rgb_B_T_H_W_D = x_rgb_B_T_H_W_D + self.modality_embed_rgb
        x_skel_B_T_H_W_D = x_skel_B_T_H_W_D + self.modality_embed_skel

        T = x_rgb_B_T_H_W_D.shape[1]
        x_B_2T_H_W_D = torch.cat([x_rgb_B_T_H_W_D, x_skel_B_T_H_W_D], dim=1)

        # Compute the rope table ONCE for the single-stream (T,H,W) grid, then TILE it (not a
        # range-extended call with T'=2T) so that flattened position i in the RGB block and
        # position i in the skeleton block (offset by T*H*W) get bit-identical rotary codes.
        rope_emb_THW_1_1_D = self.pos_embedder(x_rgb_B_T_H_W_D, fps=fps)
        rope_emb_2THW_1_1_D = torch.cat([rope_emb_THW_1_1_D, rope_emb_THW_1_1_D], dim=0)

        return x_B_2T_H_W_D, rope_emb_2THW_1_1_D, T

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        x_skel_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        timesteps_skel_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        condition_skel_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        assert intermediate_feature_ids is None, "MinimalV1JointDiT does not support intermediate_feature_ids"
        assert isinstance(data_type, DataType), f"Expected DataType, got {type(data_type)}"

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
            x_skel_B_C_T_H_W = torch.cat(
                [x_skel_B_C_T_H_W, condition_skel_input_mask_B_C_T_H_W.type_as(x_skel_B_C_T_H_W)], dim=1
            )
        else:
            B, _, T_, H_, W_ = x_B_C_T_H_W.shape
            zero_mask_rgb = torch.zeros((B, 1, T_, H_, W_), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
            zero_mask_skel = torch.zeros(
                (B, 1, T_, H_, W_), dtype=x_skel_B_C_T_H_W.dtype, device=x_skel_B_C_T_H_W.device
            )
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, zero_mask_rgb], dim=1)
            x_skel_B_C_T_H_W = torch.cat([x_skel_B_C_T_H_W, zero_mask_skel], dim=1)

        x_B_2T_H_W_D, rope_emb_2THW_1_1_D, T = self.prepare_joint_embedded_sequence(
            x_B_C_T_H_W, x_skel_B_C_T_H_W, fps=fps, padding_mask=padding_mask,
        )

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set if img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            if timesteps_skel_B_T.ndim == 1:
                timesteps_skel_B_T = timesteps_skel_B_T.unsqueeze(1)
            timesteps_B_2T = torch.cat(
                [timesteps_B_T * self.timestep_scale, timesteps_skel_B_T * self.timestep_scale], dim=1
            )
            t_embedding_B_2T_D, adaln_lora_B_2T_3D = self.t_embedder(timesteps_B_2T)
            t_embedding_B_2T_D = self.t_embedding_norm(t_embedding_B_2T_D)

        for block in self.blocks:
            x_B_2T_H_W_D = block(
                x_B_2T_H_W_D,
                t_embedding_B_2T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_2THW_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_2T_3D,
                extra_per_block_pos_emb=None,
            )

        x_rgb_B_T_H_W_D = x_B_2T_H_W_D[:, :T]
        x_skel_B_T_H_W_D = x_B_2T_H_W_D[:, T:]
        t_embedding_rgb_B_T_D = t_embedding_B_2T_D[:, :T]
        t_embedding_skel_B_T_D = t_embedding_B_2T_D[:, T:]
        adaln_lora_rgb_B_T_3D = adaln_lora_B_2T_3D[:, :T] if adaln_lora_B_2T_3D is not None else None
        adaln_lora_skel_B_T_3D = adaln_lora_B_2T_3D[:, T:] if adaln_lora_B_2T_3D is not None else None

        out_rgb_B_T_H_W_O = self.head_rgb(
            x_rgb_B_T_H_W_D, t_embedding_rgb_B_T_D, adaln_lora_B_T_3D=adaln_lora_rgb_B_T_3D
        )
        out_skel_B_T_H_W_O = self.head_skel(
            x_skel_B_T_H_W_D, t_embedding_skel_B_T_D, adaln_lora_B_T_3D=adaln_lora_skel_B_T_3D
        )

        out_rgb_B_C_T_H_W = self.unpatchify(out_rgb_B_T_H_W_O)
        out_skel_B_C_T_H_W = self.unpatchify(out_skel_B_T_H_W_O)

        return out_rgb_B_C_T_H_W, out_skel_B_C_T_H_W

    def fully_shard(self, mesh, **fsdp_kwargs):
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)

        fully_shard(self.head_rgb, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.head_skel, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        if self.extra_per_block_abs_pos_emb:
            fully_shard(self.extra_pos_embedder, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)
        fully_shard(self.x_embedder, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.skel_embedder, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        if self.extra_image_context_dim is not None:
            fully_shard(self.img_context_proj, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)
