"""OSCAR diffusers wrapper.

Public API:

    from oscar_diffusers import OSCARDiffusersPipeline

    pipe = OSCARDiffusersPipeline.from_pretrained(
        "masterwu/OSCAR-2B",                    # HF Hub repo
        # or "/path/to/local/checkpoints",      # local checkpoint dir
    )
    out = pipe(
        first_frame="path/to/first_frame.png",
        skeleton_video="path/to/gripper_scenario.mp4",
        prompt="robot grasps the bottle",
        start_frame=91,
        num_inference_steps=35,
        guidance_scale=6.0,
        seed=42,
    )
    # out.frames is a list of np.uint8 (H, W, 3) arrays.

For the 14 benchmark cases shipped on Hugging Face, use the assets helper:

    out = OSCARDiffusersPipeline.from_assets(
        "masterwu/OSCAR-2B", case="agibot_465",
    )(seed=42)

The wrapper delegates all real compute to ``inference._core.run_inference``;
no model code is duplicated here.
"""
from oscar_diffusers.pipeline import OSCARDiffusersPipeline, OSCARPipelineOutput

__all__ = ["OSCARDiffusersPipeline", "OSCARPipelineOutput"]
