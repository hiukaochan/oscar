#!/usr/bin/env bash
# OSCAR public inference dispatcher.
# Usage: bash scripts/run_inference.sh <case>
#
# Available cases (run with no arg to see the list):
#   agibot_465              agibot_360
#   droid_TRI               droid_AUTOLab_1202
#   airoa_ep000593          airoa_ep000719
#   airoa_moma_ep000755     airoa_moma_ep000963
#   interndata_ep936        interndata_ep1185
#   rh20t_cfg5_103          rh20t_cfg5_0004
#   rh20t_cfg7_34           rh20t_cfg7_0002

set -euo pipefail

case_name="${1:-}"
if [ -z "${case_name}" ]; then
    cat >&2 <<USAGE
Usage: $0 <case>

Available cases:
  agibot_465              agibot_360
  droid_TRI               droid_AUTOLab_1202
  airoa_ep000593          airoa_ep000719
  airoa_moma_ep000755     airoa_moma_ep000963
  interndata_ep936        interndata_ep1185
  rh20t_cfg5_103          rh20t_cfg5_0004
  rh20t_cfg7_34           rh20t_cfg7_0002
USAGE
    exit 1
fi

# Hardcoded case -> start_frame map (from pick_best_eval_start at staging time).
declare -A START_FRAME=(
    [agibot_465]=91
    [agibot_360]=5
    [droid_TRI]=116
    [droid_AUTOLab_1202]=57
    [airoa_ep000593]=240
    [airoa_ep000719]=327
    [airoa_moma_ep000755]=46
    [airoa_moma_ep000963]=149
    [interndata_ep936]=266
    [interndata_ep1185]=0
    [rh20t_cfg5_103]=83
    [rh20t_cfg5_0004]=95
    [rh20t_cfg7_34]=308
    [rh20t_cfg7_0002]=127
)

# Per-case seed map. Each case uses a distinct seed so that running multiple
# cases in a single session does not collide with PRNG state, and the values
# match the reference evaluation protocol (idx 0..13 -> seed 42..55).
declare -A SEED=(
    [agibot_465]=42
    [agibot_360]=43
    [droid_TRI]=44
    [droid_AUTOLab_1202]=45
    [airoa_ep000593]=46
    [airoa_ep000719]=47
    [airoa_moma_ep000755]=48
    [airoa_moma_ep000963]=49
    [interndata_ep936]=50
    [interndata_ep1185]=51
    [rh20t_cfg5_103]=52
    [rh20t_cfg5_0004]=53
    [rh20t_cfg7_34]=54
    [rh20t_cfg7_0002]=55
)

if [ -z "${START_FRAME[$case_name]+_}" ]; then
    echo "error: unknown case '$case_name'" >&2
    exit 2
fi

SF="${START_FRAME[$case_name]}"
SEED_VAL="${SEED[$case_name]:-42}"
ASSET="checkpoints/assets/${case_name}"
if [ ! -d "$ASSET" ]; then
    echo "error: assets not found at $ASSET (did you run 'hf download <handle>/OSCAR-2B --local-dir checkpoints/'?)" >&2
    exit 3
fi

# Extract first frame from rgb.mp4 at start_frame.
first_frame="/tmp/${case_name}_first_$$.png"
.venv/bin/python - "$ASSET/rgb.mp4" "$first_frame" "$SF" <<'PY'
import sys
import imageio.v3 as iio
from PIL import Image
rgb_path, out_path, sf = sys.argv[1], sys.argv[2], int(sys.argv[3])
v = iio.imread(rgb_path, plugin='FFMPEG')
Image.fromarray(v[sf]).save(out_path)
PY

# Extract prompt from caption.pickle.
prompt="$(.venv/bin/python - "$ASSET/caption.pickle" <<'PY'
import pickle
import sys
c = pickle.load(open(sys.argv[1], 'rb'))
if isinstance(c, str):
    print(c)
elif isinstance(c, dict) and 'caption' in c:
    print(c['caption'])
else:
    print(str(c))
PY
)"

mkdir -p outputs
# OSCAR_QA_RGB_VIDEO=1 feeds the GT rgb.mp4 as batch["video"] for byte-parity
# with worldsim_private/scripts/evaluate.py. Production users without GT must
# leave this unset; the script then tiles --first-frame across the window.
qa_args=()
if [ "${OSCAR_QA_RGB_VIDEO:-0}" = "1" ]; then
    qa_args+=(--rgb-video "$ASSET/rgb.mp4")
fi
PYTHONPATH=. \
    .venv/bin/torchrun --nproc_per_node=1 inference/inference_oscar.py \
        --checkpoint checkpoints \
        --first-frame "$first_frame" \
        --skeleton-video "$ASSET/gripper_scenario.mp4" \
        --start-frame "$SF" \
        --prompt "$prompt" \
        --num-steps 35 --guidance 6.0 --seed "$SEED_VAL" \
        --output "outputs/${case_name}.mp4" \
        "${qa_args[@]}"

rm -f "$first_frame"
echo "[done] outputs/${case_name}.mp4"
