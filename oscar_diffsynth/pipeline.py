"""OSCAR DiffSynth-Studio pipeline.

Mirrors ``WanVideoPipeline`` / ``HunyuanVideoPipeline`` from
DiffSynth-Studio in shape: ``from_dcp`` constructor + ``ModelManager``
integration + ``__call__`` with the same kwargs the diffusers wrapper
uses. Internally it delegates to ``inference._core.run_inference`` so the
output is byte-identical at L0..L6 to both ``inference_oscar.py`` and the
diffusers wrapper.

The DiffSynth-Studio upstream has no Cosmos-Predict2.5 loader. The
``from_dcp`` path here therefore reuses worldsim's
``load_model_from_checkpoint`` directly — exactly the same model object
the diffusers wrapper builds. ``from_model_manager`` is reserved for
future deployments where DiffSynth manages the text encoder and VAE
separately; right now it is a thin compatibility shim that defers to
``from_dcp``.
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
class OSCARDiffSynthOutput:
    """Decoded RGB video. ``frames`` is a list of ``np.uint8`` arrays of
    shape ``(H, W, 3)``, mirroring the diffusers wrapper's output type so
    downstream code can swap pipelines without changing the call site."""

    frames: list[np.ndarray]


def _default_config_file() -> str:
    return "worldsim/_src/configs/agibot_control/config.py"


class OSCARDiffSynthPipeline:
    """OSCAR inference with a DiffSynth-Studio-style API.

    DiffSynth-Studio's pipelines are organized as ``Pipeline(model_manager)``
    where ``ModelManager`` aggregates pre-loaded components. OSCAR doesn't
    fit that mold — the worldsim model wrapper attaches text encoder,
    scheduler, and tokenizer in one shot via ``load_model_from_checkpoint``
    — so the canonical constructor here is :meth:`from_dcp`.
    """

    def __init__(self, model, repo_root: Path):
        self.model = model
        self.repo_root = repo_root

    @classmethod
    def from_dcp(
        cls,
        repo_id_or_dir: str,
        *,
        experiment_name: str = "cosmos2_robot_plus_human_v2_70f",
        config_file: str | None = None,
        enable_fsdp: bool = False,
    ) -> "OSCARDiffSynthPipeline":
        """Load directly from a distributed-checkpoint directory (or HF
        Hub repo id). Equivalent to the diffusers wrapper's
        ``from_pretrained`` — both call ``load_model_from_checkpoint``
        under the hood, so the resulting model object is bit-identical."""
        setup_backends()

        import torch.distributed as dist
        import worldsim._ext.imaginaire.utils.distributed as ws_dist
        if not dist.is_initialized():
            ws_dist.init()

        from worldsim._src.utils.model_loader import load_model_from_checkpoint

        p = Path(repo_id_or_dir)
        if p.is_dir():
            repo_root = p
        else:
            from huggingface_hub import snapshot_download
            repo_root = Path(snapshot_download(repo_id=repo_id_or_dir, repo_type="model"))

        oscar_pkg_root = Path(__file__).resolve().parent.parent
        os.chdir(oscar_pkg_root)

        model, _ = load_model_from_checkpoint(
            experiment_name=experiment_name,
            checkpoint_path=str(repo_root),
            enable_fsdp=enable_fsdp,
            config_file=config_file or _default_config_file(),
        )
        return cls(model=model, repo_root=repo_root)

    @classmethod
    def from_model_manager(
        cls, model_manager, repo_id_or_dir: str, **kwargs,
    ) -> "OSCARDiffSynthPipeline":
        """Reserved for a future deployment where DiffSynth-Studio owns the
        text encoder + VAE registration. Today the upstream loader has no
        Cosmos-Predict2.5 / DCP reader, so this is a thin shim that
        forwards to :meth:`from_dcp` and ignores ``model_manager``. Users
        who need the manager-mediated path should subclass and override.
        """
        del model_manager  # see docstring
        return cls.from_dcp(repo_id_or_dir, **kwargs)

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
    ) -> OSCARDiffSynthOutput:
        """Run inference. Argument shape matches the diffusers wrapper —
        the two wrappers are interchangeable at the call site."""
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
        frames_01 = (sample.float().clamp(-1, 1).cpu()[0] + 1.0) / 2.0
        frames_uint8 = (
            frames_01.permute(1, 2, 3, 0).clamp(0, 1) * 255
        ).to(torch.uint8).numpy()
        return OSCARDiffSynthOutput(frames=list(frames_uint8))

    def from_assets(self, case: str) -> "_AssetsCaseRunner":
        """Bind one of the 14 packaged benchmark cases. Returns a callable
        that reuses the loaded pipeline; intended for notebook one-liners.
        Mirrors ``OSCARDiffusersPipeline.from_assets`` so notebooks can
        swap libraries by changing one import line."""
        return _AssetsCaseRunner(self, case)


class _AssetsCaseRunner:
    """Run a packaged benchmark case using its bundled start_frame +
    prompt + skeleton path."""

    def __init__(self, pipe: OSCARDiffSynthPipeline, case: str):
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
        cap = pickle.loads((self.asset_dir / "caption.pickle").read_bytes())
        self.prompt = (
            cap["caption"] if isinstance(cap, dict) and "caption" in cap
            else cap if isinstance(cap, str) else str(cap)
        )

    @torch.no_grad()
    def __call__(self, **overrides: Any) -> OSCARDiffSynthOutput:
        kwargs = dict(
            first_frame=self._extract_first_frame(),
            skeleton_video=self.asset_dir / "gripper_scenario.mp4",
            prompt=self.prompt,
            start_frame=self.start_frame,
        )
        kwargs.update(overrides)
        return self.pipe(**kwargs)

    def _extract_first_frame(self) -> np.ndarray:
        return load_video_np(
            self.asset_dir / "rgb.mp4", self.start_frame, 1, 480, 640,
        )[0]
