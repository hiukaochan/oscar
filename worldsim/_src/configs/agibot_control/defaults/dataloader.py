# SPDX-License-Identifier: Apache-2.0
"""Dataloader registration stub (inference-only release).

Training dataloaders are not shipped with the public OSCAR inference release.
We still register placeholder nodes for every ``data_train`` group name that
experiment configs override to -- otherwise Hydra's defaults-list resolution
fails when constructing the experiment config. Each registered node is an
empty ``{}`` (no instantiation will occur for inference).
"""

from hydra.core.config_store import ConfigStore


# Names referenced by ``defaults: [{"override /data_train": ...}]`` across
# the agibot_control experiment configs in this release.
_DATA_TRAIN_PLACEHOLDER_NAMES = [
    "agibot_video_20260125",
    "agibot_action_20260307",
    "unified_robot_20260326",
    "embodiment_v1",
    "embodiment_v1_mesh",
    "embodiment_v1_verify",
    "embodiment_agibot_only",
    "gh200_embodiment_v1",
    "gh200_embodiment_v1_mesh",
    "embodiment_equal_weights",
    "cosmos2_robot_plus_human_v2",
    "cosmos2_robot_plus_human_v2_70f",
]


def agibot_register_dataloader():
    """Register data_train placeholder nodes for inference-only release."""
    cs = ConfigStore.instance()
    for name in _DATA_TRAIN_PLACEHOLDER_NAMES:
        cs.store(
            group="data_train",
            package="dataloader_train",
            name=name,
            node={},
        )
