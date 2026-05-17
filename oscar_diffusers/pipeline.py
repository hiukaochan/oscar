"""OSCAR diffusers-style pipeline.

Thin wrapper around ``inference._core.run_inference``. Mirrors the diffusers
pipeline API surface (``from_pretrained`` + ``__call__`` returning a
named-tuple output) without inheriting ``diffusers.DiffusionPipeline`` —
the underlying worldsim model is not a stock UNet/transformer that
``DiffusionPipeline.save_pretrained`` knows how to serialize, so subclassing
would force us to fight the diffusers loader instead of using
``load_model_from_checkpoint`` (the only path that matches the training-time
numerics; see ``docs/WRAPPERS_PLAN.md`` gotcha #1).
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference._core import (
    COSMOS_DEFAULT_NEGATIVE_PROMPT,
    load_first_frame_np,
    load_video_np,
    run_inference,
    setup_backends,
)


@dataclass
class OSCARPipelineOutput:
    """Decoded RGB video.

    ``frames`` is a list of ``np.uint8`` arrays of shape ``(H, W, 3)``, one
    per output frame. Use ``imageio.mimsave`` (or any standard mp4 encoder)
    to persist; the wrapper itself does not write files — that decision is
    left to the caller so notebook users can keep tensors in memory and
    deployment scripts can pick their codec.
    """

    frames: list[np.ndarray]


def _default_config_file() -> str:
    return "worldsim/_src/configs/agibot_control/config.py"


def _resolve_repo(repo_id_or_dir: str) -> Path:
    """Accept either a local directory or a HF Hub repo id. Returns the
    on-disk root that contains ``model/`` + (optionally) ``assets/`` +
    ``case_map.json``."""
    p = Path(repo_id_or_dir)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=repo_id_or_dir, repo_type="model")
    return Path(local)


class OSCARDiffusersPipeline:
    """OSCAR inference pipeline with a diffusers-style API.

    Construct via :meth:`from_pretrained` so the worldsim model loader sees
    a real on-disk DCP checkpoint. Calling the instance runs the same
    text-encode → DiT-35-step → VAE-decode pipeline that
    ``inference/inference_oscar.py`` runs, byte-identical at L0..L6 (see
    ``docs/WRAPPERS_PLAN.md``).
    """

    def __init__(self, model, repo_root: Path):
        self.model = model
        self.repo_root = repo_root

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_dir: str,
        *,
        experiment_name: str = "cosmos2_robot_plus_human_v2_70f",
        config_file: str | None = None,
        enable_fsdp: bool = False,
    ) -> "OSCARDiffusersPipeline":
        """Build the pipeline from a local directory or HF Hub repo.

        The ``repo_id_or_dir`` argument must point at the layout produced by
        the OSCAR release (``model/__0_0.distcp`` + ``.metadata`` at the
        root, optionally with ``assets/`` and ``case_map.json``). For HF
        Hub ids the snapshot is downloaded to the standard
        ``huggingface_hub`` cache directory.
        """
        setup_backends()

        # Single-rank torch.distributed init — ``load_model_from_checkpoint``
        # expects ``dist.get_world_size()`` to work.
        import torch.distributed as dist
        import worldsim._ext.imaginaire.utils.distributed as ws_dist
        if not dist.is_initialized():
            ws_dist.init()

        from worldsim._src.utils.model_loader import load_model_from_checkpoint

        repo_root = _resolve_repo(repo_id_or_dir)

        # The imaginaire config helper expects a relative path that resolves
        # against PYTHONPATH; chdir into the package root so
        # ``worldsim/_src/configs/...`` is importable as a dotted module.
        oscar_pkg_root = Path(__file__).resolve().parent.parent
        os.chdir(oscar_pkg_root)

        model, _ = load_model_from_checkpoint(
            experiment_name=experiment_name,
            checkpoint_path=str(repo_root),
            enable_fsdp=enable_fsdp,
            config_file=config_file or _default_config_file(),
        )
        return cls(model=model, repo_root=repo_root)

    @torch.no_grad()
    def __call__(
        self,
        *,
        first_frame: str | Path | np.ndarray,
        skeleton_video: str | Path,
        prompt: str,
        negative_prompt: str | None = None,
        rgb_video: str | Path | None = None,
        start_frame: int = 0,
        num_inference_steps: int = 35,
        guidance_scale: float = 6.0,
        shift: float = 5.0,
        num_frames: int = 81,
        height: int = 480,
        width: int = 640,
        fps: float | None = None,
        seed: int = 42,
    ) -> OSCARPipelineOutput:
        """Run inference. ``first_frame`` may be a path or a pre-decoded
        ``(H, W, 3)`` ``uint8`` array. ``skeleton_video`` is the path to
        ``gripper_scenario.mp4`` (NOT ``mesh_scenario.mp4``).

        ``rgb_video`` is optional and only useful for byte-parity checks
        against the worldsim oracle metric — production users with a
        single still image should leave it ``None`` (the wrapper will tile
        the first frame across the 81-frame window).
        """
        if isinstance(first_frame, np.ndarray):
            assert first_frame.dtype == np.uint8 and first_frame.shape[-1] == 3
            first_frame_np = first_frame
            if first_frame_np.shape[0] != height or first_frame_np.shape[1] != width:
                import cv2
                first_frame_np = cv2.resize(
                    first_frame_np, (width, height), interpolation=cv2.INTER_LINEAR,
                )
        else:
            first_frame_np = load_first_frame_np(Path(first_frame), height, width)

        skel_np = load_video_np(
            Path(skeleton_video), start_frame, num_frames, height, width,
        )

        if rgb_video is not None:
            rgb_np = load_video_np(
                Path(rgb_video), start_frame, num_frames, height, width,
            )
        else:
            rgb_np = np.tile(first_frame_np[None], (num_frames, 1, 1, 1))

        if fps is None:
            import imageio.v3 as iio
            try:
                fps = float(iio.immeta(str(skeleton_video), plugin="FFMPEG")["fps"])
            except Exception:
                fps = 15.0

        sample = run_inference(
            self.model,
            rgb_frames=rgb_np,
            condition_frames=skel_np,
            prompt=prompt,
            negative_prompt=negative_prompt,
            fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            num_steps=num_inference_steps,
            guidance=guidance_scale,
            shift=shift,
            seed=seed,
        )
        # (1, 3, T, H, W) in [-1, 1] fp32 → list of (H, W, 3) uint8.
        frames_01 = ((sample.float().clamp(-1, 1).cpu()[0] + 1.0) / 2.0)
        frames_uint8 = (frames_01.permute(1, 2, 3, 0).clamp(0, 1) * 255).to(torch.uint8).numpy()
        return OSCARPipelineOutput(frames=list(frames_uint8))

    def from_assets(self, case: str) -> "_AssetsCaseRunner":
        """Bind one of the 14 packaged benchmark cases. Returns a callable
        that re-uses the loaded pipeline; intended for notebook one-liners.
        """
        return _AssetsCaseRunner(self, case)


class _AssetsCaseRunner:
    """Run a packaged benchmark case using its bundled start_frame + prompt
    + skeleton path. Wraps :meth:`OSCARDiffusersPipeline.__call__`."""

    def __init__(self, pipe: OSCARDiffusersPipeline, case: str):
        self.pipe = pipe
        self.case = case
        case_map_path = pipe.repo_root / "case_map.json"
        if not case_map_path.exists():
            raise FileNotFoundError(
                f"case_map.json missing at {pipe.repo_root}; this pipeline "
                "was not initialized from an OSCAR-2B release directory."
            )
        case_map = json.loads(case_map_path.read_text())
        if case not in case_map:
            raise KeyError(
                f"unknown case {case!r}; available: {sorted(case_map)}"
            )
        self.info = case_map[case]
        self.asset_dir = pipe.repo_root / "assets" / case
        self.start_frame = self.info.get("start_frame", 0)
        # Caption pickle: usually a plain str, sometimes a dict.
        cap = pickle.loads((self.asset_dir / "caption.pickle").read_bytes())
        self.prompt = (
            cap["caption"] if isinstance(cap, dict) and "caption" in cap
            else cap if isinstance(cap, str) else str(cap)
        )

    @torch.no_grad()
    def __call__(self, **overrides: Any) -> OSCARPipelineOutput:
        kwargs = dict(
            first_frame=self._extract_first_frame(),
            skeleton_video=self.asset_dir / "gripper_scenario.mp4",
            prompt=self.prompt,
            start_frame=self.start_frame,
        )
        kwargs.update(overrides)
        return self.pipe(**kwargs)

    def _extract_first_frame(self) -> np.ndarray:
        """Pull frame ``start_frame`` from rgb.mp4 via the same decord/cv2
        path the CLI uses, so the first frame is byte-identical to what
        the dispatcher would feed."""
        return load_video_np(
            self.asset_dir / "rgb.mp4", self.start_frame, 1, 480, 640,
        )[0]
