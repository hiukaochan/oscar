#!/usr/bin/env python3
"""Minimal skeleton-overlay demo for a single DROID (Franka Panda + Robotiq 2F-85) episode.

This is a self-contained, CPU-only illustration of how OSCAR turns recorded
robot proprioception into the 2D kinematic-skeleton conditioning signal used by
the world model. For one episode it:

  1. reads the robot state  (joint angles, end-effector pose, gripper openness)
  2. solves link positions  (forward kinematics through the URDF)
  3. projects them to pixels (world -> camera -> image, using the recorded
     camera intrinsics / extrinsics)
  4. draws the skeleton + gripper and overlays it on the RGB frame

It deliberately hard-codes the Franka + Robotiq configuration and avoids the
HuggingFace download / mesh-rendering / multi-arm machinery of the full
pipeline so the whole thing is one readable file.

Usage:
    python demo/render_skeleton.py
    python demo/render_skeleton.py --sample demo/sample --out out --max-frames 60

Dependencies: numpy, opencv-python, yourdfpy, imageio[ffmpeg]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import yourdfpy

# ---------------------------------------------------------------------------
# Robot configuration: Franka Emika Panda (7-DoF) + Robotiq 2F-85 gripper.
# Mirrors the "franka_robotiq" entry of the full pipeline's URDF registry.
# ---------------------------------------------------------------------------
URDF_FILE = "panda_robotiq.urdf"

# Joint names that the dataset's joint_angles columns map onto, in order.
ARM_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]

# Kinematic chain to draw, from end-effector link down to the base link.
# Lines are drawn between consecutive links; the first entry is the EE link
# whose full pose anchors the gripper drawing.
CHAIN_LINKS = [
    "robotiq_arg2f_base_link",
    "panda_link8", "panda_link7", "panda_link6", "panda_link5",
    "panda_link4", "panda_link3", "panda_link2", "panda_link1", "panda_link0",
]

# Colors are RGB tuples drawn directly onto RGB frames (imageio order).
SKELETON_COLOR = (255, 255, 0)   # yellow arm links
JOINT_COLOR = (0, 0, 255)        # blue joint dots
FINGER_COLOR = (255, 0, 0)       # red gripper fingers
X_AXIS_COLOR = (0, 0, 255)       # gripper frame: X blue
Y_AXIS_COLOR = (0, 255, 0)       #                Y green
Z_AXIS_COLOR = (255, 0, 0)       #                Z red

LINE_THICKNESS = 3
JOINT_RADIUS = 4
AXIS_LENGTH = 0.2                # meters
GRIPPER_THICKNESS = 5


# ---------------------------------------------------------------------------
# Step 2: forward kinematics
# ---------------------------------------------------------------------------
def forward_kinematics(urdf, joint_angles: np.ndarray):
    """Set joint angles, then return link positions in the URDF base frame.

    Returns:
        positions: (N, 3) array of CHAIN_LINKS positions in base frame
        ee_pose:   (4, 4) full pose of the end-effector link (CHAIN_LINKS[0])
    """
    cfg = {
        name: float(joint_angles[i])
        for i, name in enumerate(ARM_JOINT_NAMES)
        if i < len(joint_angles) and name in urdf.actuated_joint_names
    }
    urdf.update_cfg(cfg)

    graph = urdf.scene.graph
    base_link = list(graph.nodes)[0]  # scene-graph root == robot base (panda_link0)

    positions, ee_pose = [], None
    for idx, link in enumerate(CHAIN_LINKS):
        tf = graph.get(frame_from=base_link, frame_to=link)[0]  # 4x4 base->link
        positions.append(tf[:3, 3])
        if idx == 0:
            ee_pose = np.asarray(tf, dtype=np.float64)
    return np.asarray(positions), ee_pose


# ---------------------------------------------------------------------------
# Step 3: projection (world/base -> camera -> image pixels)
# ---------------------------------------------------------------------------
def project(points_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    """Project (N, 3) base-frame points to (N, 2) pixel coords.

    extrinsic is the 4x4 base-to-camera (world-to-camera) transform.
    Points behind the camera are pushed to a tiny positive Z so the bone
    direction toward the image edge is preserved instead of being dropped.
    """
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    pts_cam = (R @ points_world.T + t.reshape(3, 1)).T  # (N, 3)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    eps = 1e-3
    z = np.where(pts_cam[:, 2] >= eps, pts_cam[:, 2], eps)
    u = fx * (pts_cam[:, 0] / z) + cx
    v = fy * (pts_cam[:, 1] / z) + cy
    return np.stack([u, v], axis=-1)


# ---------------------------------------------------------------------------
# Step 4: drawing
# ---------------------------------------------------------------------------
def _clipped_line(img, p0, p1, color, thickness, arrow=False):
    """Draw a line/arrow clipped to the image rectangle. p0/p1 are (u, v)."""
    h, w = img.shape[:2]
    clamp = 100_000
    pt0 = (int(round(np.clip(p0[0], -clamp, clamp))), int(round(np.clip(p0[1], -clamp, clamp))))
    pt1 = (int(round(np.clip(p1[0], -clamp, clamp))), int(round(np.clip(p1[1], -clamp, clamp))))
    ok, a, b = cv2.clipLine((0, 0, w, h), pt0, pt1)
    if not ok or a == b:
        return
    if arrow:
        cv2.arrowedLine(img, a, b, color, thickness, tipLength=0.15)
    else:
        cv2.line(img, a, b, color, thickness)


def draw_skeleton(img, positions, extrinsic, intrinsic):
    """Draw arm links + joint dots. Drops near-zero-length segments."""
    # Filter consecutive duplicate positions (fixed joints w/ zero offset).
    uniq = [positions[0]]
    for p in positions[1:]:
        if np.linalg.norm(p - uniq[-1]) >= 1e-4:
            uniq.append(p)
    if len(uniq) < 2:
        return
    uv = project(np.asarray(uniq), extrinsic, intrinsic)
    for i in range(len(uv) - 1):
        _clipped_line(img, uv[i], uv[i + 1], SKELETON_COLOR, LINE_THICKNESS)
    h, w = img.shape[:2]
    for u, v in uv:
        if -1e5 < u < 1e5 and -1e5 < v < 1e5:
            cv2.circle(img, (int(round(u)), int(round(v))), JOINT_RADIUS, JOINT_COLOR, -1)


def draw_gripper(img, ee_pose, extrinsic, intrinsic, openness):
    """Draw EE coordinate axes + two gripper fingers anchored at the EE pose."""
    rot, pos = ee_pose[:3, :3], ee_pose[:3, 3]
    # Fingers spread along the EE Y axis (panda_hand convention), opening
    # angle scaled by gripper openness in [0, 1].
    s = AXIS_LENGTH * np.sin(np.pi / 12 * openness)
    c = AXIS_LENGTH * np.cos(np.pi / 12 * openness)
    local = np.array([
        [0, 0, 0],
        [AXIS_LENGTH, 0, 0],   # X
        [0, AXIS_LENGTH, 0],   # Y
        [0, 0, AXIS_LENGTH],   # Z
        [0, s, c],             # left finger
        [0, -s, c],            # right finger
    ], dtype=np.float64)
    world = (rot @ local.T + pos.reshape(3, 1)).T  # (6, 3)
    uv = project(world, extrinsic, intrinsic)
    _clipped_line(img, uv[0], uv[1], X_AXIS_COLOR, GRIPPER_THICKNESS, arrow=True)
    _clipped_line(img, uv[0], uv[2], Y_AXIS_COLOR, GRIPPER_THICKNESS, arrow=True)
    _clipped_line(img, uv[0], uv[3], Z_AXIS_COLOR, GRIPPER_THICKNESS, arrow=True)
    _clipped_line(img, uv[0], uv[4], FINGER_COLOR, GRIPPER_THICKNESS, arrow=True)
    _clipped_line(img, uv[0], uv[5], FINGER_COLOR, GRIPPER_THICKNESS, arrow=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def render_episode(sample_dir: Path, urdf_path: Path, out_dir: Path,
                   max_frames: int | None, fps: float):
    meta = np.load(sample_dir / "episode_meta.npz", allow_pickle=False)
    joint_angles = meta["joint_angles"]          # (T, 7)
    ee_pose = meta["ee_pose"]                     # (T, 4, 4)  (unused for FK, kept for reference)
    gripper = meta["gripper_openness"]            # (T,)
    intrinsic = meta["camera_intrinsic"]          # (3, 3)
    extrinsic = meta["camera_extrinsic"]          # (4, 4), static camera
    is_per_frame = bool(meta["camera_is_per_frame"])

    urdf = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_w = iio.imopen(out_dir / "overlay.mp4", "w", plugin="pyav")
    overlay_w.init_video_stream("libx264", fps=fps)
    skeleton_w = iio.imopen(out_dir / "skeleton.mp4", "w", plugin="pyav")
    skeleton_w.init_video_stream("libx264", fps=fps)

    n = 0
    for t, frame in enumerate(iio.imiter(sample_dir / "rgb.mp4")):
        if max_frames is not None and t >= max_frames:
            break
        ext = extrinsic[t] if is_per_frame else extrinsic
        intr = intrinsic[t] if (is_per_frame and intrinsic.ndim == 3) else intrinsic

        # 1) read state -> 2) FK -> 3) project -> 4) draw
        positions, fk_ee = forward_kinematics(urdf, joint_angles[t])
        overlay = frame.copy()
        skel = np.zeros_like(frame)
        for canvas in (overlay, skel):
            draw_skeleton(canvas, positions, ext, intr)
            draw_gripper(canvas, fk_ee, ext, intr, float(gripper[t]))

        overlay_w.write_frame(np.ascontiguousarray(overlay))
        skeleton_w.write_frame(np.ascontiguousarray(skel))
        n += 1

    overlay_w.close()
    skeleton_w.close()
    print(f"Rendered {n} frames -> {out_dir/'overlay.mp4'} , {out_dir/'skeleton.mp4'}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, default=here / "sample",
                    help="dir with episode_meta.npz + rgb.mp4")
    ap.add_argument("--urdf", type=Path, default=here / "assets" / URDF_FILE)
    ap.add_argument("--out", type=Path, default=here.parent / "out")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()
    render_episode(args.sample, args.urdf, args.out, args.max_frames, args.fps)


if __name__ == "__main__":
    main()
