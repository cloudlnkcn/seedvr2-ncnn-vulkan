#!/usr/bin/env bash
# Official checkpoint -> locked converter -> native package verification.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
kind=${1:-image}
output=${2:-.cache/user-models/image}
binary=${3:-dist/seedvr2/bin/seedvr2}
if [[ "$kind" != image && "$kind" != video ]]; then
  echo 'Usage: bash tools/convert_models.sh image|video OUTPUT NATIVE_BINARY' >&2; exit 2
fi
command -v uv >/dev/null || { echo 'Install uv and the native build prerequisites listed in docs/TUTORIAL.md.' >&2; exit 2; }
[[ -x "$binary" ]] || { echo "Native executable not found: $binary" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Choose a new output directory: $output" >&2; exit 2; }
# Keep exports and packages on the same filesystem for hard-link assembly.
mkdir -p "$(dirname "$output")"
work=$(mktemp -d "$(dirname "$output")/conversion.XXXXXXXX")
echo "Conversion intermediates and log: $work"
exec > >(tee "$work/conversion.log") 2>&1
uv venv --python 3.13 "$work/venv"
py="$work/venv/bin/python"
uv pip install --python "$py" torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$py" -r tools/export-requirements.lock.txt
python3 tools/prepare_models.py
python3 tools/prepare_pnnx.py --jobs 2
"$py" tools/export_dit_block.py --pnnx .deps/bin/pnnx --blocks 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 --output "$work/dit"
"$py" tools/export_vae_image.py --pnnx .deps/bin/pnnx --output "$work/vae-image"
image_output="$output"
if [[ "$kind" == video ]]; then image_output="$work/image"; fi
"$py" tools/export_image_package.py --pnnx .deps/bin/pnnx --blocks "$work/dit" --vae "$work/vae-image" --output "$image_output"
"$binary" models verify --kind image --model "$image_output" > "$work/image-verify.json"
if [[ "$kind" == video ]]; then
  "$py" tools/export_vae_video.py --pnnx .deps/bin/pnnx --output "$work/vae-video"
  "$py" tools/export_video_package.py --image-package "$image_output" --vae "$work/vae-video" --output "$output"
  "$binary" models verify --kind video --model "$output" > "$work/video-verify.json"
fi
echo "Verified package: $output"
echo 'Keep the program and model folder for offline inference. Conversion logs and intermediates have been retained.'
