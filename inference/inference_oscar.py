#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""User-facing OSCAR inference entry.

Loads the OSCAR Cosmos-I2V-Control model via the existing
``worldsim`` codepath, constructs a single-sample ``data_batch``
from a user-provided first frame + skeleton video + prompt,
runs ``generate_samples_from_batch`` exactly as the oracle would,
and writes the decoded RGB video.

Unlike the training-side oracle (``inference_agibot_control.py``),
this script does NOT instantiate the training dataloader and does
NOT load pre-baked cached negative-prompt embeddings. Text embeddings
for both the prompt and the negative prompt are computed online via
the model's registered ``Cosmos-Reason1-7B`` text encoder.

Usage example (run from the oscar repo root):

    PYTHONPATH=. \\
    HF_HUB_OFFLINE=1 \\
    .venv/bin/torchrun --nproc_per_node=1 \\
        inference/inference_oscar.py \\
        --checkpoint checkpoints/model \\
        --first-frame /path/to/ff.png \\
        --skeleton-video /path/to/gripper_scenario.mp4 \\
        --prompt "robot grasps the bottle" \\
        --negative-prompt "" \\
        --output out.mp4

For typical usage prefer ``bash scripts/run_inference.sh <case>``, which
wraps this entry point with the per-case ``start_frame``/seed map.

The actual compute happens in ``inference/_core.py`` — both this CLI and
the ``oscar_diffusers`` / ``oscar_diffsynth`` wrappers delegate there so
all three surfaces share one byte-identical pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange

import worldsim._ext.imaginaire.utils.distributed  # noqa: F401
from worldsim._ext.imaginaire.utils import log
from worldsim._ext.imaginaire.visualize.video import save_img_or_video
from worldsim._src.utils.model_loader import load_model_from_checkpoint

from inference._core import (
    load_first_frame_np,
    load_video_np,
    run_inference,
    setup_backends,
)


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OSCAR public inference entry")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a DCP iter_<N> directory (or a .pt/.pth consolidated ckpt).",
    )
    p.add_argument("--first-frame", type=Path, required=True)
    p.add_argument("--skeleton-video", type=Path, required=True)
    p.add_argument(
        "--rgb-video",
        type=Path,
        default=None,
        help=(
            "Optional GT video used as batch['video']. The model conditions only "
            "on frame 0, but the Wan2.1 VAE has a temporal receptive field that "
            "spans the first chunk, so encoding 16 tiled identical frames vs the "
            "real first 16 frames produces a slightly different conditioning "
            "latent. Production users with only a first frame should omit this "
            "(the script will tile --first-frame across the window). For "
            "metric-parity verification against worldsim_private's "
            "scripts/evaluate.py, pass the GT mp4 here so batch['video'] matches "
            "the oracle byte-for-byte."
        ),
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help=(
            "Seek into --skeleton-video by this many frames before reading the "
            "81-frame window. Oracle scripts/evaluate.py uses "
            "pick_best_eval_start() to pick a gripper-centered window (often "
            "frame 91 for agibot samples). Pass that value here to reproduce "
            "the oracle benchmark numbers."
        ),
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help=(
            "Negative prompt for classifier-free guidance. If omitted, the "
            "oracle's COSMOS_DEFAULT_NEGATIVE_PROMPT (a long quality-degrading "
            "description string) is used, matching scripts/evaluate.py."
        ),
    )
    p.add_argument("--num-steps", type=int, default=35)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)  # matches oracle prepare_batch H=480, W=640
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("out.mp4"),
        help="Output path; the suffix is stripped and ``.mp4`` re-appended by save_img_or_video.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Per-sample fps fed to the model (matches the training dataloader's "
            "fps tensor and the output mp4 framerate). If omitted, the skeleton "
            "mp4's intrinsic fps is read via imageio.immeta -- this matches the "
            "training collate which records real-valued mp4 fps."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_arguments()
    setup_backends()

    # Init torch.distributed (single-rank).
    worldsim._ext.imaginaire.utils.distributed.init()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    log.info(f"world_size: {world_size}")

    # The imaginaire config helper converts the config_file path -> dotted
    # module name with ``path.replace('/', '.')``. That requires the path to
    # be relative to a directory on ``sys.path`` (and have no leading slash).
    # The repo root must be on ``PYTHONPATH``; we chdir to it so a relative
    # path "worldsim/_src/configs/agibot_control/config.py" resolves to the
    # importable module ``worldsim._src.configs.agibot_control.config``.
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    config_file = "worldsim/_src/configs/agibot_control/config.py"
    model, _ = load_model_from_checkpoint(
        experiment_name="cosmos2_robot_plus_human_v2_70f",
        checkpoint_path=args.checkpoint,
        enable_fsdp=False,
        config_file=config_file,
    )

    first_frame_np = load_first_frame_np(args.first_frame, args.height, args.width)
    skel_np = load_video_np(
        args.skeleton_video, args.start_frame, args.num_frames, args.height, args.width,
    )
    if args.rgb_video is not None:
        rgb_np = load_video_np(
            args.rgb_video, args.start_frame, args.num_frames, args.height, args.width,
        )
    else:
        rgb_np = np.tile(first_frame_np[None], (args.num_frames, 1, 1, 1))

    if args.fps is None:
        try:
            meta = iio.immeta(str(args.skeleton_video), plugin="FFMPEG")
            args.fps = float(meta["fps"])
            log.info(f"auto-detected skeleton fps={args.fps}")
        except Exception as e:
            log.warning(f"could not auto-detect fps ({e!r}); falling back to 15.0")
            args.fps = 15.0

    sample = run_inference(
        model,
        rgb_frames=rgb_np,
        condition_frames=skel_np,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        fps=args.fps,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_steps=args.num_steps,
        guidance=args.guidance,
        shift=args.shift,
        seed=args.seed,
    )

    # Save as MP4 via the imaginaire helper (handles ffmpeg + bitrate).
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = str(args.output).split(".mp4")[0]
    save_img_or_video(
        rearrange(sample.float().clamp(-1, 1).cpu(), "b c t h w -> c t h (b w)") * 0.5
        + 0.5,
        base,
        fps=args.fps,
    )
    log.info(f"==> finished save video to {base}.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
