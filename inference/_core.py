"""Reusable OSCAR inference primitives.

``inference_oscar.py`` is the CLI entry point; the two library wrappers
``oscar_diffusers`` and ``oscar_diffsynth`` reach into this module instead
of duplicating the data-batch / text-embed / generate / decode pipeline.

Everything in here mirrors ``worldsim_private/scripts/evaluate.py`` so that
the public release stays byte-identical to the training-time oracle on
L0..L6 (text-encoder hidden state through final sampled latent). The only
permitted divergence is the bf16 conv3d cross-process noise in the VAE
decoder (~1 ULP at ~0.7 dB PSNR on motion-heavy cases); see
docs/WRAPPERS_PLAN.md gotcha #9.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from worldsim._ext.imaginaire.utils import misc
from inference._constants import COSMOS_DEFAULT_NEGATIVE_PROMPT

__all__ = [
    "COSMOS_DEFAULT_NEGATIVE_PROMPT",
    "setup_backends",
    "load_first_frame_np",
    "load_video_np",
    "prepare_batch_joint",
    "run_inference",
]


_VAE_TEMPORAL_STRIDE = 4
_NORM_IMAGE = transforms.Normalize(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True,
)


def setup_backends() -> None:
    """Match worldsim_private/scripts/evaluate.py lines 41 + 1683 + 1702."""
    torch.backends.cuda.preferred_linalg_library(backend="cusolver")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.enable_grad(False)


def load_first_frame_np(path: Path, height: int, width: int) -> np.ndarray:
    """Decode the first frame at ``(height, width)`` uint8 (H, W, 3) RGB."""
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    return np.array(img)


def load_video_np(
    path: Path, start_frame: int, num_frames: int, height: int, width: int,
) -> np.ndarray:
    """Read ``num_frames`` frames starting at ``start_frame``, matching
    ``read_video_frames`` from worldsim's scripts/evaluate.py byte-for-byte:
    decord-first decoding with a cv2 VideoCapture fallback, freeze-frame
    tail-pad for short episodes, and cv2.resize / cv2.INTER_LINEAR. Using
    imageio + PIL.BILINEAR diverges from the oracle metric by 1 uint8 level
    per pixel on non-target-resolution sources.
    """
    v: np.ndarray | None = None
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(path))
        total = len(vr)
        if start_frame >= total:
            raise ValueError(
                f"video {path} has {total} frames, start={start_frame} out of range"
            )
        end = min(total, start_frame + num_frames)
        v = vr.get_batch(list(range(start_frame, end))).asnumpy()
    except Exception:
        cap = cv2.VideoCapture(str(path))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"could not read frames from {path}")
        v = np.stack(frames)

    if v.shape[0] < num_frames:
        pad = np.repeat(v[-1:], num_frames - v.shape[0], axis=0)
        v = np.concatenate([v, pad], axis=0)
    v = v[:num_frames]

    if v.shape[1] != height or v.shape[2] != width:
        v = np.stack([
            cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)
            for f in v
        ])
    return v


def _to_norm_video_tensor(frames_np: np.ndarray, T: int, H: int, W: int) -> torch.Tensor:
    """uint8 (T, H, W, 3) -> normalized float (1, 3, T, H, W) in [-1, 1].

    Shared by ``prepare_batch_joint`` (inference: single frame tiled across T)
    and the training-side droid dataset collate (full video windows).
    """
    video = torch.stack([
        torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
    ])
    B, _, C, _, _ = video.shape
    video = video.permute(0, 2, 1, 3, 4)
    video = _NORM_IMAGE(video.reshape(B * C, T, H, W).permute(1, 0, 2, 3))
    video = video.permute(1, 0, 2, 3).reshape(B, C, T, H, W)
    return video


def prepare_batch_joint(
    first_frame_rgb: np.ndarray,
    first_frame_skeleton: np.ndarray,
    caption: str,
    *,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
) -> dict:
    """Joint-generation single-sample batch: only the first RGB frame and
    first skeleton frame are real conditioning signal — both get tiled across
    the full ``num_frames`` window (the model only ever reads/overwrites frame
    0 via its per-stream conditioning mask; frames 1..T-1 are targets to be
    generated, not oracle input). Both ``first_frame_rgb`` and
    ``first_frame_skeleton`` must be uint8 (H, W, 3).
    """
    T = num_frames
    H, W = height, width
    latent_T = 1 + (T - 1) // _VAE_TEMPORAL_STRIDE

    rgb_tiled = np.tile(first_frame_rgb[None], (T, 1, 1, 1))
    skel_tiled = np.tile(first_frame_skeleton[None], (T, 1, 1, 1))

    videos = _to_norm_video_tensor(rgb_tiled, T, H, W)
    conds = _to_norm_video_tensor(skel_tiled, T, H, W)

    return {
        "video": videos,
        "hint_key": conds,
        "is_preprocessed": True,
        "ai_caption": [caption],
        "t5_text_embeddings": torch.zeros(1, 512, 4096),
        "t5_text_mask": torch.zeros(1, 512),
        "num_frames": T,
        "image_size": torch.tensor([H, W]),
        "fps": torch.tensor([float(fps)]),
        "padding_mask": torch.zeros(1, 1, H, W),
        "frame_valid_mask": torch.ones(1, latent_T),
    }


def run_inference(
    model,
    *,
    first_frame_rgb: np.ndarray,
    first_frame_skeleton: np.ndarray,
    prompt: str,
    negative_prompt: str | None = None,
    fps: float = 15.0,
    num_frames: int = 81,
    height: int = 480,
    width: int = 640,
    num_steps: int = 35,
    guidance: float = 6.0,
    shift: float = 5.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """End-to-end OSCAR joint inference. Returns ``(rgb_video, skeleton_video)``,
    each ``(1, 3, T, H, W)`` fp32 in ``[-1, 1]``.

    Inputs ``first_frame_rgb`` and ``first_frame_skeleton`` must be uint8
    ``(H, W, 3)`` at the model's training resolution. Use ``load_first_frame_np``
    to build them in the same way the CLI does, or pass arbitrary arrays for
    non-file inputs.
    """
    assert getattr(model, "text_encoder", None) is not None, (
        "model.text_encoder is None; this experiment expected compute_online=True."
    )
    torch.manual_seed(seed)

    data_batch = prepare_batch_joint(
        first_frame_rgb=first_frame_rgb,
        first_frame_skeleton=first_frame_skeleton,
        caption=prompt,
        num_frames=num_frames,
        fps=fps,
        height=height,
        width=width,
    )
    data_batch = misc.to(data_batch, **model.tensor_kwargs)

    embed_dtype = model.tensor_kwargs.get("dtype", torch.bfloat16)
    neg_text = (
        negative_prompt
        if negative_prompt is not None
        else COSMOS_DEFAULT_NEGATIVE_PROMPT
    )
    # Neg-first encode order mirrors evaluate.py (line 1214 before per-sample
    # loop). Reversing it shifts cuDNN's algorithm-selection cache and
    # perturbs bf16 reductions by ~1e-3.
    neg_emb = model.text_encoder.compute_text_embeddings_online(
        {"ai_caption": [neg_text], "images": None}, "ai_caption",
    )
    cond_emb = model.text_encoder.compute_text_embeddings_online(
        {"ai_caption": data_batch["ai_caption"], "images": None}, "ai_caption",
    )
    data_batch["t5_text_embeddings"] = cond_emb.to(dtype=embed_dtype)
    data_batch["t5_text_mask"] = torch.ones(
        cond_emb.shape[0], cond_emb.shape[1], device="cuda", dtype=embed_dtype,
    )
    data_batch["neg_t5_text_embeddings"] = neg_emb.to(dtype=embed_dtype)
    data_batch["neg_t5_text_mask"] = data_batch["t5_text_mask"]

    raw_data, x0, condition = model.get_data_and_condition(data_batch)
    sample_rgb, sample_skel = model.generate_samples_from_batch(
        data_batch,
        guidance=guidance,
        seed=seed,
        state_shape=x0.shape[1:],
        n_sample=x0.shape[0],
        num_steps=num_steps,
        is_negative_prompt=True,
        shift=shift,
    )
    return model.decode(sample_rgb), model.decode(sample_skel)
