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

"""Joint RGB + skeleton condition/conditioner.

Skeleton moves from being a clean oracle side-channel (``ControlVideo2WorldCondition``
/ ``i2v_WAN2PT1_cond_latents``) to a second co-generated target, on equal footing
with RGB. Unlike RGB (which can be conditioned on a random number of leading
frames), skeleton is always conditioned on exactly its first frame.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from einops import rearrange
from torch.distributed import get_process_group_ranks

from worldsim._src.conditioner.cosmos2_v2v_conditioner import Video2WorldCondition
from worldsim._src.modules.conditioner import GeneralConditioner, T2VCondition
from worldsim._src.utils.context_parallel import broadcast_split_tensor, find_split


@dataclass(frozen=True)
class JointVideo2WorldCondition(Video2WorldCondition):
    gt_frames_skel: Optional[torch.Tensor] = None
    condition_skel_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None

    def set_joint_video_condition(
        self,
        gt_frames: torch.Tensor,
        gt_frames_skel: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
        conditional_frames_probs: Optional[Dict[int, float]] = None,
        num_conditional_frames_skel: int = 1,
    ) -> "JointVideo2WorldCondition":
        """Sets both RGB and skeleton video conditioning. RGB reuses the existing
        (possibly randomized) frame-count logic unchanged; skeleton is always
        conditioned on exactly its first ``num_conditional_frames_skel`` frame(s).
        """
        _rgb_condition = self.set_video_condition(
            gt_frames=gt_frames,
            random_min_num_conditional_frames=random_min_num_conditional_frames,
            random_max_num_conditional_frames=random_max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=conditional_frames_probs,
        )
        kwargs = _rgb_condition.to_dict(skip_underscore=False)

        B, _, T, H, W = gt_frames_skel.shape
        condition_skel_input_mask_B_C_T_H_W = torch.zeros(
            B, 1, T, H, W, dtype=gt_frames_skel.dtype, device=gt_frames_skel.device
        )
        if T > 1:  # image batches (T==1) get an all-zero mask, mirroring Video2WorldCondition
            condition_skel_input_mask_B_C_T_H_W[:, :, :num_conditional_frames_skel, :, :] = 1

        kwargs["gt_frames_skel"] = gt_frames_skel
        kwargs["condition_skel_input_mask_B_C_T_H_W"] = condition_skel_input_mask_B_C_T_H_W
        return type(self)(**kwargs)

    def edit_for_inference(
        self,
        is_cfg_conditional: bool = True,
        num_conditional_frames: int = 1,
        num_conditional_frames_skel: int = 1,
    ) -> "JointVideo2WorldCondition":
        _condition = self.set_joint_video_condition(
            gt_frames=self.gt_frames,
            gt_frames_skel=self.gt_frames_skel,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
            num_conditional_frames_skel=num_conditional_frames_skel,
        )
        if not is_cfg_conditional:
            # Do not use classifier-free guidance on conditional frames (matches
            # Video2WorldCondition.edit_for_inference's rationale).
            _condition.use_video_condition.fill_(True)
        return _condition

    def broadcast(self, process_group: Optional[torch.distributed.ProcessGroup]) -> "JointVideo2WorldCondition":
        if self.is_broadcasted:
            return self
        gt_frames = self.gt_frames
        gt_frames_skel = self.gt_frames_skel
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        condition_skel_input_mask_B_C_T_H_W = self.condition_skel_input_mask_B_C_T_H_W

        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["gt_frames_skel"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        kwargs["condition_skel_input_mask_B_C_T_H_W"] = None
        new_condition = T2VCondition.broadcast(type(self)(**kwargs), process_group)

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames.shape
        if process_group is not None:
            cp_ranks = get_process_group_ranks(process_group)
            cp_size = len(cp_ranks)
            use_spatial_split = (
                cp_size > condition_video_input_mask_B_C_T_H_W.shape[2]
                or condition_video_input_mask_B_C_T_H_W.shape[2] % cp_size != 0
            )
            after_split_shape = (
                find_split(condition_video_input_mask_B_C_T_H_W.shape, cp_size) if use_spatial_split else None
            )

            if T > 1 and process_group.size() > 1:
                if use_spatial_split:
                    condition_video_input_mask_B_C_T_H_W = rearrange(
                        condition_video_input_mask_B_C_T_H_W, "b c t h w -> b c (t h w)"
                    )
                    condition_skel_input_mask_B_C_T_H_W = rearrange(
                        condition_skel_input_mask_B_C_T_H_W, "b c t h w -> b c (t h w)"
                    )
                    gt_frames = rearrange(gt_frames, "b c t h w -> b c (t h w)")
                    gt_frames_skel = rearrange(gt_frames_skel, "b c t h w -> b c (t h w)")

                gt_frames = broadcast_split_tensor(gt_frames, seq_dim=2, process_group=process_group)
                gt_frames_skel = broadcast_split_tensor(gt_frames_skel, seq_dim=2, process_group=process_group)
                condition_video_input_mask_B_C_T_H_W = broadcast_split_tensor(
                    condition_video_input_mask_B_C_T_H_W, seq_dim=2, process_group=process_group
                )
                condition_skel_input_mask_B_C_T_H_W = broadcast_split_tensor(
                    condition_skel_input_mask_B_C_T_H_W, seq_dim=2, process_group=process_group
                )
                if use_spatial_split:
                    condition_video_input_mask_B_C_T_H_W = rearrange(
                        condition_video_input_mask_B_C_T_H_W,
                        "b c (t h w) -> b c t h w",
                        t=after_split_shape[0],
                        h=after_split_shape[1],
                    )
                    condition_skel_input_mask_B_C_T_H_W = rearrange(
                        condition_skel_input_mask_B_C_T_H_W,
                        "b c (t h w) -> b c t h w",
                        t=after_split_shape[0],
                        h=after_split_shape[1],
                    )
                    gt_frames = rearrange(
                        gt_frames, "b c (t h w) -> b c t h w", t=after_split_shape[0], h=after_split_shape[1]
                    )
                    gt_frames_skel = rearrange(
                        gt_frames_skel, "b c (t h w) -> b c t h w", t=after_split_shape[0], h=after_split_shape[1]
                    )
        kwargs["gt_frames"] = gt_frames
        kwargs["gt_frames_skel"] = gt_frames_skel
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        kwargs["condition_skel_input_mask_B_C_T_H_W"] = condition_skel_input_mask_B_C_T_H_W
        return type(self)(**kwargs)


class JointVideo2WorldConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> JointVideo2WorldCondition:
        output = super()._forward(batch, override_dropout_rate)
        return JointVideo2WorldCondition(**output)
