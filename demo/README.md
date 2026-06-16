# Skeleton Rendering Demo

A minimal, self-contained example of how OSCAR turns recorded robot
proprioception into the **2D kinematic-skeleton conditioning** signal that the
world model is trained on.

It renders a single [DROID](https://droid-dataset.github.io/) episode
(Franka Emika Panda + Robotiq 2F-85 gripper) and overlays the projected
skeleton on the original RGB frames.

<div align="center">
  <img src="../docs/teaser/droid.gif" width="480"/>
</div>

## What it does

For each frame, [`render_skeleton.py`](render_skeleton.py) runs four steps:

1. **Read robot state** — joint angles, end-effector pose, gripper openness
   from `episode_meta.npz`.
2. **Forward kinematics** — solve each link's position by driving the URDF with
   the joint angles (`yourdfpy`, no meshes loaded).
3. **Project** — map the 3D link positions to pixels using the recorded camera
   intrinsics + extrinsics.
4. **Draw & overlay** — draw the arm skeleton, joint dots, and gripper
   coordinate frame, composited onto the RGB frame.

It is deliberately kept to one ~250-line file with the Franka + Robotiq
configuration hard-coded. There is no HuggingFace download, mesh rendering,
multi-arm handling, or GPU/EGL dependency — everything runs on CPU in seconds.

## Run

```bash
pip install -r demo/requirements.txt
python demo/render_skeleton.py
```

Outputs are written to `out/`:

- `overlay.mp4` — skeleton drawn on top of the RGB video
- `skeleton.mp4` — skeleton only on black (the conditioning representation)

Options:

```bash
python demo/render_skeleton.py \
    --sample demo/sample \   # dir with episode_meta.npz + rgb.mp4
    --out out \              # output dir
    --max-frames 60 \        # cap frames (default: all)
    --fps 15                 # output video fps
```

## Files

```
demo/
  render_skeleton.py        # the demo (read state -> FK -> project -> overlay)
  requirements.txt
  assets/
    panda_robotiq.urdf      # Franka Panda + Robotiq 2F-85 URDF (XML only)
  sample/
    episode_meta.npz        # one DROID episode's robot state + camera params
    rgb.mp4                 # the corresponding RGB clip (90 frames)
```

## `episode_meta.npz` format

| key | shape | meaning |
|-----|-------|---------|
| `joint_angles` | `(T, 7)` | Panda arm joint angles (radians) |
| `ee_pose` | `(T, 4, 4)` | end-effector pose in robot base frame |
| `gripper_openness` | `(T,)` | normalized gripper openness `[0, 1]` |
| `camera_intrinsic` | `(3, 3)` | pinhole intrinsics matched to the RGB resolution |
| `camera_extrinsic` | `(4, 4)` | base-to-camera (world-to-camera) transform |
| `camera_is_per_frame` | scalar bool | `True` if intrinsics/extrinsics vary per frame |

To render your own episode, point `--sample` at a directory with the same
`episode_meta.npz` + `rgb.mp4` layout.
