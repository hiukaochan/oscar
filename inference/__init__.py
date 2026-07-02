"""OSCAR public-inference package.

The user-facing entry point is ``inference_oscar.py`` (CLI). The reusable
primitives live in ``_core``; both this CLI and the wrapper packages
``oscar_diffusers`` / ``oscar_diffsynth`` import from there so all three
surfaces share the same byte-identical pipeline.
"""
from inference._core import (
    COSMOS_DEFAULT_NEGATIVE_PROMPT,
    load_first_frame_np,
    load_video_np,
    prepare_batch_joint,
    run_inference,
    setup_backends,
)

__all__ = [
    "COSMOS_DEFAULT_NEGATIVE_PROMPT",
    "setup_backends",
    "load_first_frame_np",
    "load_video_np",
    "prepare_batch_joint",
    "run_inference",
]
