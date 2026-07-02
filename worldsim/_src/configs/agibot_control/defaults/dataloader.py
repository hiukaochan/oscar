# SPDX-License-Identifier: Apache-2.0
"""Dataloader registration (inference-only release, plus one real dataloader).

Most training dataloaders are not shipped with the public OSCAR inference
release, so we still register placeholder nodes for every ``data_train``
group name that experiment configs override to -- otherwise Hydra's
defaults-list resolution fails when constructing the experiment config. Each
placeholder node is an empty ``{}`` (no instantiation will occur for
inference).

One exception: ``droid_joint_v1`` is a real dataloader over the droid-layout
sample data at ``droid/`` (see ``worldsim._src.datasets.droid_joint_dataset``),
used to train/validate the joint RGB+skeleton model.
"""

from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from worldsim._ext.imaginaire.lazy_config import LazyCall as L
from worldsim._src.datasets.droid_joint_dataset import DroidJointEpisodeDataset, collate_joint_episode_batch


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

DroidJointDataloader = L(DataLoader)(
    dataset=L(DroidJointEpisodeDataset)(
        # "../droid" because inference_oscar.py's os.chdir(repo_root) convention (repo_root =
        # oscar/) means training/inference code runs with cwd=oscar/, while the droid/ sample
        # data lives one level up, alongside oscar/ (not inside it).
        root="../droid",
        num_frames=81,
        height=480,
        width=640,
    ),
    batch_size=1,
    collate_fn=collate_joint_episode_batch,
    shuffle=True,
    num_workers=0,
)


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
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="droid_joint_v1",
        node=DroidJointDataloader,
    )
