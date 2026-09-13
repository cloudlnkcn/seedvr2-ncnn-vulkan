#!/usr/bin/env bash
# Official checkpoint -> locked converter -> native package verification.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
usage() {
  cat <<'HELP'
Usage: bash tools/convert_models.sh [--check] image|video OUTPUT NATIVE_BINARY
Downloads pinned official checkpoints and converts a native FP32-B package.
--check: validate prerequisites and paths without downloads or writes.
--help: show this help. Relative paths are resolved from the repository root.
An existing output is never overwritten. Intermediates/logs are retained on failure.
Conversion does not resume automatically. See docs/LOCAL-CONVERSION.md.
HELP
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
check=false
if [[ "${1:-}" == --check ]]; then check=true; shift; fi
[[ $# == 3 ]] || { usage >&2; exit 2; }
kind=$1
output=$2
binary=$3
[[ "$kind" == image || "$kind" == video ]] || { usage >&2; exit 2; }
for command in uv python3 cmake ninja c++; do
  command -v "$command" >/dev/null || { echo "Missing prerequisite: $command. See docs/TUTORIAL.md." >&2; exit 2; }
done
[[ -x "$binary" ]] || { echo "Native executable not found: $binary" >&2; exit 2; }
[[ ! -e "$output" && ! -L "$output" ]] || { echo "Choose a new output directory: $output" >&2; exit 2; }
"$binary" version
python3 -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or newer is required"'
# Run before large downloads; neither probe changes files.
cmake --version | head -1
"$binary" engine self-test --backend cpu
if $check; then echo 'Preflight passed; conversion and full-model execution have not run.'; exit 0; fi
# Keep exports and packages on the same filesystem for hard-link assembly.
mkdir -p "$(dirname "$output")"
work=$(mktemp -d "$(dirname "$output")/conversion.XXXXXXXX")
echo "Conversion intermediates and log: $work"
exec > >(tee "$work/conversion.log") 2>&1
stage=environment
trap 'code=$?; echo "Conversion failed at stage: $stage (exit $code). Logs: $work/conversion.log" >&2; exit "$code"' ERR
uv venv --python 3.13 "$work/venv"
py="$work/venv/bin/python"
uv pip install --python "$py" torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$py" -r tools/export-requirements.lock.txt
stage=official-download
python3 tools/prepare_models.py
stage=pnnx-build
python3 tools/prepare_pnnx.py --jobs 2 --python "$py"
stage=dit-export
"$py" tools/export_dit_block.py --pnnx .deps/bin/pnnx --blocks 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 --output "$work/dit"
stage=image-vae-export
"$py" tools/export_vae_image.py --pnnx .deps/bin/pnnx --output "$work/vae-image"
image_output="$output"
if [[ "$kind" == video ]]; then image_output="$work/image"; fi
stage=image-package
"$py" tools/export_image_package.py --pnnx .deps/bin/pnnx --blocks "$work/dit" --vae "$work/vae-image" --output "$image_output"
"$binary" models verify --kind image --model "$image_output" > "$work/image-verify.json"
if [[ "$kind" == video ]]; then
  stage=video-vae-export
  "$py" tools/export_vae_video.py --pnnx .deps/bin/pnnx --output "$work/vae-video"
  stage=video-package
  "$py" tools/export_video_package.py --image-package "$image_output" --vae "$work/vae-video" --output "$output"
  "$binary" models verify --kind video --model "$output" > "$work/video-verify.json"
fi
echo "Verified package: $output"
echo 'Keep the program and model folder for offline inference. Conversion logs and intermediates have been retained.'
