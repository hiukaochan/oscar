"""OSCAR DiffSynth-Studio wrapper.

Public API:

    from oscar_diffsynth import OSCARDiffSynthPipeline

    # Direct DCP-checkpoint load (the only currently-supported path; the
    # upstream DiffSynth-Studio loader has no native Cosmos-Predict2.5 /
    # DCP reader as of this writing — see docs/WRAPPERS_PLAN.md).
    pipe = OSCARDiffSynthPipeline.from_dcp("/path/to/checkpoints")

    out = pipe(
        first_frame="path/to/first_frame.png",
        skeleton_video="path/to/gripper_scenario.mp4",
        prompt="robot grasps the bottle",
        start_frame=91,
        num_inference_steps=35,
        guidance_scale=6.0,
        seed=42,
    )

If you maintain a DiffSynth-Studio ``ModelManager`` that already wraps the
Cosmos-Reason1-7B text encoder and the Wan2.1 VAE, see
:meth:`OSCARDiffSynthPipeline.from_model_manager` for the integration
shape. That path is optional; the DCP loader is sufficient on its own.

The wrapper delegates all real compute to ``inference._core.run_inference``;
no model code is duplicated here.
"""
from oscar_diffsynth.pipeline import (
    OSCARDiffSynthPipeline,
    OSCARDiffSynthOutput,
)

__all__ = ["OSCARDiffSynthPipeline", "OSCARDiffSynthOutput"]
