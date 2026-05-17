#!/usr/bin/env bash
# Download the 8 teaser mp4s from the OSCAR project page and convert each
# to a small, looping GIF for embedding in README.md. Idempotent: rerunning
# overwrites docs/teaser/<short>.gif with a fresh build.
#
# Source mp4s are 1920x480 side-by-side (skeleton | generated | GT) at
# 5 seconds; the output is 640x160 (each panel ≈ 213 px wide) at 12 fps,
# ≈ 250 KB per GIF.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p docs/teaser

BASE="https://wuzy2115.github.io/oscar-project-page/static/videos/section_1_teaser"

CASES=(
  "agibot:robot__agibot__787_924471_a11_1290_1561__head"
  "airoa:robot__airoa_moma__ep_004153__head"
  "droid:robot__droid__REAL__success__2023-06-23__Fri_Jun_23_16:44:47_2023__20540549"
  "interndata:robot__interndata__ep158__head"
  "rh20t_cfg5:robot__rh20t_cfg5__task_0123_user_0010_scene_0009_cfg_0005__036422060215"
  "rh20t_cfg7:robot__rh20t_cfg7__task_0002_user_0014_scene_0002_cfg_0007__104122060811"
  "egodex:human__egodex__20348__head"
  "epic:human__vitra_epic__epic_kitchens_P01_05_ep_000166__head"
)

for entry in "${CASES[@]}"; do
  short="${entry%%:*}"
  case_id="${entry#*:}"
  mp4="docs/teaser/$short.mp4"
  gif="docs/teaser/$short.gif"

  echo "[$short] downloading"
  curl -fsSL "$BASE/$case_id/comparison.mp4" -o "$mp4"

  echo "[$short] converting to gif"
  ffmpeg -y -loglevel error \
    -i "$mp4" \
    -vf "scale=640:160:flags=lanczos,fps=12" \
    -loop 0 \
    "$gif"
  rm -f "$mp4"

  printf "[%-12s] %s\n" "$short" "$(du -h "$gif" | cut -f1)"
done

echo
echo "[done] 8 GIFs written to docs/teaser/"
du -sh docs/teaser/
