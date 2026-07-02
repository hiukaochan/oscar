# SPDX-License-Identifier: Apache-2.0
"""OSCAR joint RGB+skeleton generation experiment: cosmos2_joint_rgb_skel_70f.

Copy-adapted from ``v2_with_human_70f.py`` (the original single-channel
oracle-skeleton-conditioning experiment), swapping in the joint net/model/
conditioner registered in ``defaults/net.py`` / ``defaults/model.py`` /
``defaults/conditioner.py`` and the real ``droid_joint_v1`` dataloader.

No trained checkpoint exists for this architecture: ``head_skel`` and
``skel_embedder`` have no analog in the existing OSCAR checkpoint to
warm-start from (the fusion mechanism changed from weighted-sum to
temporal-axis concatenation with independent heads), so this needs training
from scratch. This experiment config documents what an external training
harness would need to override; it is not itself runnable as an inference
experiment (unlike ``v2_with_human_70f.py``, which loads a real checkpoint).
"""
import os

from hydra.core.config_store import ConfigStore

from worldsim._ext.imaginaire.lazy_config import LazyDict
from worldsim._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy


Cosmos2pt5_joint_rgb_skel_70f = LazyDict(
    dict(
        defaults=[
            {"override /net": "cosmos_v1_2B_joint"},
            {"override /model": "fsdp_cosmos_i2v_rectified_flow_joint"},
            {"override /conditioner": "joint_video_prediction_conditioner"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            {"override /ckpt_type": "dcp"},
            {"override /checkpoint": "local"},
            {"override /data_train": "droid_joint_v1"},
            "_self_",
        ],
        job=dict(group="oscar_public", name="cosmos2_joint_rgb_skel_70f"),
        model=dict(
            config=dict(
                ema=dict(enabled=False),
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs={1: 1.0},
                conditional_frame_timestep=0.1,
                use_kerras_sigma_at_inference=True,
                fsdp_shard_size=1,
                resolution="480",
                state_t=24,
                shift=5,
                use_dynamic_shift=False,
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    timestep_scale=0.001,
                    sac_config=dict(mode="predict2_2b_720_aggressive"),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    use_wan_fp32_strategy=True,
                ),
                conditioner=dict(
                    use_video_condition=dict(dropout_rate=0.0),
                    text=dict(dropout_rate=0.0, use_empty_string=False),
                ),
                tokenizer=dict(temporal_window=16),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(
                        EmbeddingConcatStrategy.FULL_CONCAT
                    ),
                    compute_online=True,
                    ckpt_path=os.environ.get(
                        "COSMOS_REASON_PATH", "nvidia/Cosmos-Reason1-7B"
                    ),
                    model_config=dict(
                        model_config=dict(
                            attn_implementation="sdpa",
                            attn_implementation_autoset=False,
                            vision_config=dict(
                                attn_implementation="sdpa",
                                attn_implementation_autoset=False,
                            ),
                        ),
                    ),
                ),
            )
        ),
    ),
    flags={"allow_objects": True},
)


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=Cosmos2pt5_joint_rgb_skel_70f["job"]["name"],
    node=Cosmos2pt5_joint_rgb_skel_70f,
)
