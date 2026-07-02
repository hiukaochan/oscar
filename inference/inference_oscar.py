#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""User-facing OSCAR joint RGB+skeleton inference entry.

Loads the OSCAR joint model (``CosmosJointRGBSkelModel``) via the existing
``worldsim`` codepath, constructs a single-sample ``data_batch`` from a
user-provided first RGB frame + first skeleton frame + prompt, runs
``generate_samples_from_batch`` to jointly denoise both streams, and writes
the two decoded videos (RGB and skeleton).

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
        --first-frame-skeleton /path/to/skeleton_ff.png \\
        --prompt "robot grasps the bottle" \\
        --negative-prompt "" \\
        --output out.mp4

The actual compute happens in ``inference/_core.py`` — both this CLI and
the ``oscar_diffusers`` / ``oscar_diffsynth`` wrappers delegate there so
all three surfaces share one pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from einops import rearrange

import worldsim._ext.imaginaire.utils.distributed  # noqa: F401
from worldsim._ext.imaginaire.utils import log
from worldsim._ext.imaginaire.visualize.video import save_img_or_video
from worldsim._src.utils.model_loader import load_model_from_checkpoint

from inference._core import (
    load_first_frame_np,
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
    p.add_argument(
        "--first-frame-skeleton",
        type=Path,
        required=True,
        help=(
            "Path to a single first-frame skeleton/gripper-pose image (e.g. frame 0 "
            "of a gripper_scenario.mp4, extracted separately). The joint model only "
            "ever conditions on this one frame -- the rest of the skeleton video is "
            "generated jointly with RGB, not supplied as input."
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
        default=15.0,
        help=(
            "Per-sample fps fed to the model (matches the training dataloader's "
            "fps tensor and the output mp4 framerate). No video file is read at "
            "inference time anymore (only first frames), so this can no longer be "
            "auto-detected -- pass the value used at training time explicitly."
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

    first_frame_rgb_np = load_first_frame_np(args.first_frame, args.height, args.width)
    first_frame_skel_np = load_first_frame_np(args.first_frame_skeleton, args.height, args.width)

    sample_rgb, sample_skel = run_inference(
        model,
        first_frame_rgb=first_frame_rgb_np,
        first_frame_skeleton=first_frame_skel_np,
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
        rearrange(sample_rgb.float().clamp(-1, 1).cpu(), "b c t h w -> c t h (b w)") * 0.5
        + 0.5,
        f"{base}_rgb",
        fps=args.fps,
    )
    save_img_or_video(
        rearrange(sample_skel.float().clamp(-1, 1).cpu(), "b c t h w -> c t h (b w)") * 0.5
        + 0.5,
        f"{base}_skel",
        fps=args.fps,
    )
    log.info(f"==> finished saving videos to {base}_rgb.mp4 and {base}_skel.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
