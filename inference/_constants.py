# SPDX-License-Identifier: Apache-2.0
"""Inference-only constants for OSCAR.

Extracted from ``worldsim._src.callbacks.every_n_draw_sample_control`` so
that ``inference/_core.py`` no longer pulls in the training-time draw-
sample callback (which imports ``wandb`` and ``worldsim._ext.callbacks.
every_n_draw_sample``, both deleted in this release).
"""

COSMOS_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with "
    "no motion, motion blur, over-saturation, shaky footage, low resolution, "
    "grainy texture, pixelated images, poorly lit areas, underexposed and "
    "overexposed scenes, poor color balance, washed out colors, choppy "
    "sequences, jerky movements, low frame rate, artifacting, color banding, "
    "unnatural transitions, outdated special effects, fake elements, "
    "unconvincing visuals, poorly edited content, jump cuts, visual noise, "
    "and flickering. Overall, the video is of poor quality."
)
