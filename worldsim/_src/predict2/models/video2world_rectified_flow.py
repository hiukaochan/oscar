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
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import attrs
import torch
import tqdm
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from worldsim._ext.imaginaire.flags import INTERNAL
from worldsim._ext.imaginaire.utils import misc
from worldsim._src.modules.conditioner import DataType
from worldsim._src.conditioner.cosmos2_v2v_conditioner import Video2WorldCondition
from worldsim._src.conditioner.cosmos2_joint_conditioner import JointVideo2WorldCondition
from worldsim._src.predict2.models.denoise_prediction import DenoisePrediction
from worldsim._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)
from worldsim._src.modules.conditioner import T2VCondition as Text2WorldCondition
NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"
COSMOS_CONTROL_KEY = "i2v_WAN2PT1_cond_latents"

class ConditioningStrategy(str, Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames

    def __str__(self) -> str:
        return self.value


@attrs.define(slots=False)
class Video2WorldModelRectifiedFlowConfig(Text2WorldModelRectifiedFlowConfig):
    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    conditional_frame_timestep: float = (
        -1.0
    )  # Noise level used for conditional frames; default is -1 which will not take effective
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
    denoise_replace_gt_frames: bool = True  # Whether to denoise the ground truth frames
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.conditioning_strategy in [
            str(ConditioningStrategy.FRAME_REPLACE),
        ]


class Video2WorldModelRectifiedFlow(Text2WorldModelRectifiedFlow):
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        # generate random number of conditional frames for training
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
    ) -> DenoisePrediction:
        """
        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (Text2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            velocity prediction
        """
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )

            # Make the first few frames of x_t be the ground truth frames
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )

                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )

                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = (
                    timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T
                )  # add dimension for batch
        # forward pass through the network
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=timesteps_B_T,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        ).float()

        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        return net_output_B_C_T_H_W

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return velocity predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            cond_v = self.denoise(noise, noise_x, timestep, condition)
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        return velocity_fn

class CosmosI2VControlModel(Video2WorldModelRectifiedFlow):
    @torch.no_grad()
    def get_data_and_condition(
            self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        # generate random number of conditional frames for training
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        latent_state = self.encode(raw_state).contiguous().float()

        if COSMOS_CONTROL_KEY not in data_batch:
            # get the condition input
            hint_key = 'hint_key'
            data_batch[COSMOS_CONTROL_KEY] = self.encode(data_batch[hint_key]).contiguous().to(**self.tensor_kwargs)

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition


class CosmosJointRGBSkelModel(Video2WorldModelRectifiedFlow):
    """Joint RGB + skeleton generation model.

    Unlike ``CosmosI2VControlModel`` (where ``hint_key`` is a full skeleton video
    injected as a clean oracle-conditioning latent, never noised), here
    ``hint_key`` is a second **training target**: it is VAE-encoded, independently
    noised, and jointly denoised alongside the RGB stream by ``MinimalV1JointDiT``.
    Only the first frame of each stream stays clean, via the same first-frame
    overwrite mechanism ``Video2WorldModelRectifiedFlow.denoise()`` already uses
    for RGB, generalized to both streams.
    """

    @torch.no_grad()
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, JointVideo2WorldCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        raw_state_rgb = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        latent_state_rgb = self.encode(raw_state_rgb).contiguous().float()

        # `hint_key` is now a full skeleton video and a second training target -- assumed
        # already normalized to [-1, 1] by the batch-prep/dataloader (same convention
        # CosmosI2VControlModel already relies on for hint_key today; it is not re-normalized
        # here, only encoded).
        raw_state_skel = data_batch["hint_key"]
        latent_state_skel = self.encode(raw_state_skel).contiguous().float()

        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        condition = condition.set_joint_video_condition(
            gt_frames=latent_state_rgb.to(**self.tensor_kwargs),
            gt_frames_skel=latent_state_skel.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
            num_conditional_frames_skel=1,
        )
        return raw_state_rgb, latent_state_rgb, condition

    def denoise(
        self,
        noise: torch.Tensor,
        noise_skel: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        xt_skel_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: JointVideo2WorldCondition,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if condition.is_video:
            # --- RGB stream first-frame overwrite (identical to Video2WorldModelRectifiedFlow.denoise) ---
            condition_state_rgb = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_rgb = condition_state_rgb * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )
            xt_B_C_T_H_W = condition_state_rgb * condition_video_mask + xt_B_C_T_H_W * (1 - condition_video_mask)

            timesteps_rgb_B_T = timesteps_B_T
            if self.config.conditional_frame_timestep >= 0:
                cond_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                ts_cond_B_1_T_1_1 = torch.ones_like(cond_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                ts_B_1_T_1_1 = ts_cond_B_1_T_1_1 * cond_mask_B_1_T_1_1 + timesteps_rgb_B_T * (
                    1 - cond_mask_B_1_T_1_1
                )
                timesteps_rgb_B_T = ts_B_1_T_1_1.squeeze()
                timesteps_rgb_B_T = (
                    timesteps_rgb_B_T.unsqueeze(0) if timesteps_rgb_B_T.ndim == 1 else timesteps_rgb_B_T
                )

            # --- skeleton stream first-frame overwrite (same mechanism, independent mask) ---
            condition_state_skel = condition.gt_frames_skel.type_as(xt_skel_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_skel = condition_state_skel * 0

            condition_skel_mask = condition.condition_skel_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_skel_B_C_T_H_W
            )
            xt_skel_B_C_T_H_W = condition_state_skel * condition_skel_mask + xt_skel_B_C_T_H_W * (
                1 - condition_skel_mask
            )

            timesteps_skel_B_T = timesteps_B_T
            if self.config.conditional_frame_timestep >= 0:
                cond_mask_skel_B_1_T_1_1 = condition_skel_mask.mean(dim=[1, 3, 4], keepdim=True)
                ts_cond_skel_B_1_T_1_1 = (
                    torch.ones_like(cond_mask_skel_B_1_T_1_1) * self.config.conditional_frame_timestep
                )
                ts_skel_B_1_T_1_1 = ts_cond_skel_B_1_T_1_1 * cond_mask_skel_B_1_T_1_1 + timesteps_skel_B_T * (
                    1 - cond_mask_skel_B_1_T_1_1
                )
                timesteps_skel_B_T = ts_skel_B_1_T_1_1.squeeze()
                timesteps_skel_B_T = (
                    timesteps_skel_B_T.unsqueeze(0) if timesteps_skel_B_T.ndim == 1 else timesteps_skel_B_T
                )
        else:
            timesteps_rgb_B_T = timesteps_B_T
            timesteps_skel_B_T = timesteps_B_T

        net_output_rgb_B_C_T_H_W, net_output_skel_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            x_skel_B_C_T_H_W=xt_skel_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_rgb_B_T,
            timesteps_skel_B_T=timesteps_skel_B_T,
            **condition.to_dict(),
        )
        net_output_rgb_B_C_T_H_W = net_output_rgb_B_C_T_H_W.float()
        net_output_skel_B_C_T_H_W = net_output_skel_B_C_T_H_W.float()

        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_rgb_velocity = noise - condition.gt_frames.type_as(net_output_rgb_B_C_T_H_W)
            net_output_rgb_B_C_T_H_W = gt_rgb_velocity * condition_video_mask + net_output_rgb_B_C_T_H_W * (
                1 - condition_video_mask
            )
            gt_skel_velocity = noise_skel - condition.gt_frames_skel.type_as(net_output_skel_B_C_T_H_W)
            net_output_skel_B_C_T_H_W = gt_skel_velocity * condition_skel_mask + net_output_skel_B_C_T_H_W * (
                1 - condition_skel_mask
            )

        return net_output_rgb_B_C_T_H_W, net_output_skel_B_C_T_H_W

    def forward(self, data_batch: dict[str, torch.Tensor]) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        _, x0_rgb_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)
        x0_skel_B_C_T_H_W = condition.gt_frames_skel

        # single shared scalar time t_B, sampled once -- both channels are noised/interpolated
        # with the same `sigmas` below (spec: "single shared scalar timestep applied identically
        # to both channels").
        batch_size = x0_rgb_B_C_T_H_W.size()[0]
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")

        epsilon_rgb_B_C_T_H_W = torch.randn(x0_rgb_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
        epsilon_skel_B_C_T_H_W = torch.randn(x0_skel_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)

        x0_rgb_B_C_T_H_W, condition, epsilon_rgb_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_rgb_B_C_T_H_W, condition, epsilon_rgb_B_C_T_H_W, t_B
        )
        # NOTE: broadcast_split_for_model_parallelsim only CP-splits the RGB x0/epsilon pair (and,
        # via condition.broadcast(), gt_frames_skel/condition_skel_input_mask_B_C_T_H_W); the
        # skeleton epsilon is NOT CP-split here. For cp_size==1 (this environment) the call is a
        # no-op passthrough so this has no effect; real multi-GPU CP training needs a
        # dual-stream-aware variant of this helper -- flagged as an untested follow-up.
        x0_skel_B_C_T_H_W = condition.gt_frames_skel

        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)
        sigmas = self.rectified_flow.get_sigmas(timesteps, self.tensor_kwargs_fp32)
        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")

        xt_rgb_B_C_T_H_W, vt_rgb_B_C_T_H_W = self.rectified_flow.get_interpolation(
            epsilon_rgb_B_C_T_H_W, x0_rgb_B_C_T_H_W, sigmas
        )
        xt_skel_B_C_T_H_W, vt_skel_B_C_T_H_W = self.rectified_flow.get_interpolation(
            epsilon_skel_B_C_T_H_W, x0_skel_B_C_T_H_W, sigmas
        )

        vt_pred_rgb_B_C_T_H_W, vt_pred_skel_B_C_T_H_W = self.denoise(
            noise=epsilon_rgb_B_C_T_H_W,
            noise_skel=epsilon_skel_B_C_T_H_W,
            xt_B_C_T_H_W=xt_rgb_B_C_T_H_W.to(**self.tensor_kwargs),
            xt_skel_B_C_T_H_W=xt_skel_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
        )

        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32)

        def _masked_per_instance_loss(vt_pred: torch.Tensor, vt_gt: torch.Tensor) -> torch.Tensor:
            frame_valid_mask = data_batch.get("frame_valid_mask", None)
            T_latent = vt_pred.shape[2]
            if frame_valid_mask is not None and frame_valid_mask.shape[1] == T_latent:
                mask = frame_valid_mask.to(vt_pred.device)
                mask_B_1_T_1_1 = mask[:, None, :, None, None]
                C, H, W = vt_pred.shape[1], vt_pred.shape[3], vt_pred.shape[4]
                per_element_loss = (vt_pred - vt_gt) ** 2
                return (per_element_loss * mask_B_1_T_1_1).sum(dim=[1, 2, 3, 4]) / (
                    mask.sum(dim=1) * C * H * W
                ).clamp(min=1)
            return torch.mean((vt_pred - vt_gt) ** 2, dim=list(range(1, vt_pred.dim())))

        per_instance_loss_rgb = _masked_per_instance_loss(vt_pred_rgb_B_C_T_H_W, vt_rgb_B_C_T_H_W)
        per_instance_loss_skel = _masked_per_instance_loss(vt_pred_skel_B_C_T_H_W, vt_skel_B_C_T_H_W)
        per_instance_loss = per_instance_loss_rgb + per_instance_loss_skel  # sum of both channel losses

        loss = torch.mean(time_weights_B * per_instance_loss)
        output_batch = {
            "x0": x0_rgb_B_C_T_H_W,
            "x0_skel": x0_skel_B_C_T_H_W,
            "xt": xt_rgb_B_C_T_H_W,
            "xt_skel": xt_skel_B_C_T_H_W,
            "sigma": sigmas,
            "condition": condition,
            "model_pred": vt_pred_rgb_B_C_T_H_W,
            "model_pred_skel": vt_pred_skel_B_C_T_H_W,
            "edm_loss": loss,
            "edm_loss_rgb": torch.mean(time_weights_B * per_instance_loss_rgb),
            "edm_loss_skel": torch.mean(time_weights_B * per_instance_loss_skel),
            "timesteps": timesteps,
            "per_instance_loss": per_instance_loss,
            "n_cond_frames": condition.num_conditional_frames_B,
        }
        return output_batch, loss

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        _, x0_rgb, condition_from_data = self.get_data_and_condition(data_batch)
        x0_skel = condition_from_data.gt_frames_skel

        condition = condition.set_joint_video_condition(
            gt_frames=x0_rgb,
            gt_frames_skel=x0_skel,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
            num_conditional_frames_skel=1,
        )
        uncondition = uncondition.set_joint_video_condition(
            gt_frames=x0_rgb,
            gt_frames_skel=x0_skel,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
            num_conditional_frames_skel=1,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0_rgb, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0_rgb, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(
            noise: torch.Tensor, noise_x: torch.Tensor, noise_skel: torch.Tensor, noise_x_skel: torch.Tensor,
            timestep: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            cond_v_rgb, cond_v_skel = self.denoise(noise, noise_skel, noise_x, noise_x_skel, timestep, condition)
            uncond_v_rgb, uncond_v_skel = self.denoise(noise, noise_skel, noise_x, noise_x_skel, timestep, uncondition)
            v_rgb = cond_v_rgb + guidance * (cond_v_rgb - uncond_v_rgb)
            v_skel = cond_v_skel + guidance * (cond_v_skel - uncond_v_skel)
            return v_rgb, v_skel

        return velocity_fn

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        noise_rgb = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape), torch.float32, self.tensor_kwargs["device"], seed,
        )
        # Different seed offset so the two streams don't start from identical noise.
        noise_skel = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape), torch.float32, self.tensor_kwargs["device"], seed + 1,
        )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(
            num_steps,
            device=self.tensor_kwargs["device"],
            shift=shift,
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )
        # Independent scheduler instance for the skeleton stream: both are deterministic and
        # driven off the identical (num_steps, shift, use_kerras_sigma) schedule, so calling
        # .step() once per iteration on each keeps them in lockstep without either stream's
        # internal step-index counter interfering with the other's (which a single shared
        # scheduler stepped twice per iteration would risk).
        sample_scheduler_skel = copy.deepcopy(self.sample_scheduler)

        timesteps = self.sample_scheduler.timesteps

        velocity_fn = self.get_velocity_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        latents_rgb, latents_skel = noise_rgb, noise_skel

        if INTERNAL:
            timesteps_iter = timesteps
        else:
            timesteps_iter = tqdm.tqdm(timesteps, desc="Generating joint RGB+skeleton samples", total=len(timesteps))

        for t in timesteps_iter:
            timestep = torch.stack([t])
            v_rgb, v_skel = velocity_fn(noise_rgb, latents_rgb, noise_skel, latents_skel, timestep.unsqueeze(0))

            temp_rgb = self.sample_scheduler.step(
                v_rgb.unsqueeze(0), t, latents_rgb[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            temp_skel = sample_scheduler_skel.step(
                v_skel.unsqueeze(0), t, latents_skel[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents_rgb = temp_rgb.squeeze(0)
            latents_skel = temp_skel.squeeze(0)

        return latents_rgb, latents_skel