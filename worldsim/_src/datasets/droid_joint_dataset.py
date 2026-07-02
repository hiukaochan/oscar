# SPDX-License-Identifier: Apache-2.0
"""Real dataset loader for droid-formatted robot episodes.

Layout: ``droid/<episode_id>/[<camera_serial>/]{caption.pickle, episode_meta.npz,
gripper_scenario.mp4, rgb.mp4}``. Only items with all of ``rgb.mp4``,
``gripper_scenario.mp4``, and ``caption.pickle`` present are used -- some
episodes (e.g. single-camera demo samples) only ship ``gripper_scenario.mp4``
for inference and are silently skipped here. ``episode_meta.npz`` (joint
angles / EE pose / camera calibration) is not needed for training the joint
DiT: it was only used offline to *render* ``gripper_scenario.mp4`` in the
first place (see ``oscar/demo/render_skeleton.py``); the rendered video is
what gets trained on.

``gripper_scenario.mp4`` is now a second training **target** (co-generated
with RGB), not oracle conditioning -- see ``CosmosJointRGBSkelModel``.
"""

import pickle
from pathlib import Path
from typing import List

import imageio.v3 as iio
import torch

from inference._core import _to_norm_video_tensor, load_video_np

_VAE_TEMPORAL_STRIDE = 4


class DroidJointEpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, num_frames: int, height: int, width: int, start_frame: int = 0):
        self.root = Path(root)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.start_frame = start_frame
        # Every "rgb.mp4" found under `root` defines one item, as long as its sibling
        # "gripper_scenario.mp4" and "caption.pickle" also exist -- this naturally handles both
        # the single-camera-at-episode-root and multi-camera-subfolder layouts without
        # special-casing them.
        self.items: List[Path] = sorted(
            rgb_path.parent
            for rgb_path in self.root.rglob("rgb.mp4")
            if (rgb_path.parent / "gripper_scenario.mp4").exists()
            and (rgb_path.parent / "caption.pickle").exists()
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item_dir = self.items[idx]
        with open(item_dir / "caption.pickle", "rb") as f:
            caption = pickle.load(f)["caption"]

        rgb_np = load_video_np(
            item_dir / "rgb.mp4", self.start_frame, self.num_frames, self.height, self.width,
        )
        skel_np = load_video_np(
            item_dir / "gripper_scenario.mp4", self.start_frame, self.num_frames, self.height, self.width,
        )

        # Canonical fps = the skeleton mp4's intrinsic fps, matching the convention
        # inference_oscar.py's (now-removed) auto-fps-detection previously used.
        try:
            meta = iio.immeta(str(item_dir / "gripper_scenario.mp4"), plugin="FFMPEG")
            fps = float(meta["fps"])
        except Exception:
            fps = 15.0

        return {"rgb_np": rgb_np, "skel_np": skel_np, "caption": caption, "fps": fps}


def collate_joint_episode_batch(items: List[dict]) -> dict:
    """Stacks a list of ``DroidJointEpisodeDataset.__getitem__`` outputs into the
    batch-dict contract ``CosmosJointRGBSkelModel.get_data_and_condition()``
    expects: the same shape ``prepare_batch_joint`` produces for inference, but
    built from full rgb/skeleton video windows (training targets) instead of a
    single tiled first frame.
    """
    T = items[0]["rgb_np"].shape[0]
    H, W = items[0]["rgb_np"].shape[1:3]
    latent_T = 1 + (T - 1) // _VAE_TEMPORAL_STRIDE
    B = len(items)

    videos = torch.cat([_to_norm_video_tensor(item["rgb_np"], T, H, W) for item in items], dim=0)
    conds = torch.cat([_to_norm_video_tensor(item["skel_np"], T, H, W) for item in items], dim=0)
    captions = [item["caption"] for item in items]
    fps = torch.tensor([float(item["fps"]) for item in items])

    return {
        "video": videos,
        "hint_key": conds,
        "is_preprocessed": True,
        "ai_caption": captions,
        "t5_text_embeddings": torch.zeros(B, 512, 4096),
        "t5_text_mask": torch.zeros(B, 512),
        "num_frames": T,
        "image_size": torch.tensor([H, W]),
        "fps": fps,
        "padding_mask": torch.zeros(B, 1, H, W),
        "frame_valid_mask": torch.ones(B, latent_T),
    }
