#!/usr/bin/env bash
set -euo pipefail
build=${1:?build directory required}
install=$(realpath "${2:?install prefix required}")
evidence=${3:?evidence directory required}
mkdir -p "$evidence"
ctest --test-dir "$build" --output-on-failure > "$evidence/ctest.log" 2>&1
python3 -m unittest discover -s tests -p test_pipeline_contract.py -v > "$evidence/evidence-contract.log" 2>&1
python3 tools/check_cli.py "$install/bin/seedvr2" --output "$evidence/cli.json"
python3 tools/check_engine.py "$install/bin/seedvr2" --output "$evidence/engine-boundaries.json"
python3 tools/check_delivery.py --binary "$install/bin/seedvr2" --output "$evidence/first-use.json"
cmake -S examples/sdk -B "$build/installed-sdk" -G Ninja -DCMAKE_PREFIX_PATH="$install" > "$evidence/sdk-build.log" 2>&1
cmake --build "$build/installed-sdk" >> "$evidence/sdk-build.log" 2>&1
"$build/installed-sdk/seedvr2-sdk-example" > "$evidence/sdk-smoke.json"
icds=(/usr/share/vulkan/icd.d/lvp_icd*.json)
if [ -f "/usr/share/vulkan/icd.d/lvp_icd.$(uname -m).json" ]; then
  icds=("/usr/share/vulkan/icd.d/lvp_icd.$(uname -m).json")
fi
if [ ! -f "${icds[0]}" ]; then
  printf '{"status":"FAILED","reason":"Required Mesa Vulkan ICD is missing"}\n' > "$evidence/mesa.json"
  exit 1
fi
VK_DRIVER_FILES="${icds[0]}" VK_ICD_FILENAMES="${icds[0]}" VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  "$install/bin/seedvr2" engine self-test --backend vulkan --gpu 0 > "$evidence/mesa.json" 2> "$evidence/mesa.stderr"
if rg -i 'vuid-|validation error' "$evidence/mesa.stderr"; then exit 1; fi
VK_DRIVER_FILES="${icds[0]}" VK_ICD_FILENAMES="${icds[0]}" VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  ctest --test-dir "$build" -L vulkan --no-tests=error --output-on-failure > "$evidence/mesa-regressions.log" 2>&1
if rg -i 'vuid-|validation error' "$evidence/mesa-regressions.log"; then exit 1; fi
printf '{"scope":"Native Linux small tests and installed SDK; full 3B weights are a separate real-device protocol","full_model":"NOT_RUN_IN_CI"}\n' > "$evidence/scope.json"
