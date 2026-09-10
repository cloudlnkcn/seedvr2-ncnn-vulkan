#!/usr/bin/env bash
set -euo pipefail
build=${1:?build directory required}
install=$(realpath "${2:?install prefix required}")
evidence=$(realpath -m "${3:?evidence directory required}")
mkdir -p "$evidence"
ctest --test-dir "$build" --output-junit "$evidence/ctest.xml" --output-on-failure > "$evidence/ctest.log" 2>&1
python3 -m unittest discover -s tests -p test_pipeline_contract.py -v > "$evidence/evidence-contract.log" 2>&1
python3 tools/check_cli.py "$install/bin/seedvr2" --output "$evidence/cli.json"
python3 tools/check_engine.py "$install/bin/seedvr2" --output "$evidence/engine-boundaries.json"
python3 tools/check_delivery.py --binary "$install/bin/seedvr2" --output "$evidence/first-use.json"
cmake --fresh -S examples/sdk -B "$build/installed-sdk" -G Ninja -DCMAKE_PREFIX_PATH="$install" > "$evidence/sdk-build.log" 2>&1
cmake --build "$build/installed-sdk" >> "$evidence/sdk-build.log" 2>&1
"$build/installed-sdk/seedvr2-sdk-example" > "$evidence/sdk-smoke.json"
"$install/bin/seedvr2" engine status > "$evidence/engine-build.json"
python3 - "$evidence" <<'PY'
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
cli=json.loads((root/'engine-build.json').read_text())['implementation']
sdk=json.loads((root/'sdk-smoke.json').read_text())['implementation']
if sdk['sdk_library_sha256'] != cli['sdk_library_sha256']:
    raise SystemExit('Installed SDK example loaded a different library than the candidate CLI')
(root/'sdk-identity.json').write_text(json.dumps(dict(passed=True,
    sdk_library_sha256=sdk['sdk_library_sha256'],example_executable_sha256=sdk['executable_sha256']),indent=2)+'\n')
PY
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
if rg -i 'vuid-|validation error' "$evidence/mesa.json" "$evidence/mesa.stderr"; then exit 1; fi
VK_DRIVER_FILES="${icds[0]}" VK_ICD_FILENAMES="${icds[0]}" VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  "$install/bin/seedvr2" engine devices > "$evidence/mesa-arithmetic.json" 2> "$evidence/mesa-arithmetic.stderr"
VK_DRIVER_FILES="${icds[0]}" VK_ICD_FILENAMES="${icds[0]}" VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  ctest --test-dir "$build" -L vulkan --no-tests=error --output-junit "$evidence/mesa-ctest.xml" --output-on-failure > "$evidence/mesa-regressions.log" 2>&1
if rg -i 'vuid-|validation error' "$evidence/mesa-regressions.log"; then exit 1; fi
python3 - "$evidence" <<'PY'
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
root=Path(sys.argv[1])
def summary(path):
    rows=ET.parse(path).getroot().findall('.//testcase')
    skipped=[r.get('name') for r in rows if r.find('skipped') is not None]
    failed=[r.get('name') for r in rows if r.find('failure') is not None or r.find('error') is not None]
    return dict(total=len(rows),passed=len(rows)-len(skipped)-len(failed),skipped=skipped,failed=failed)
report=dict(scope='Native Linux small tests and installed SDK; full 3B weights require separate real-device evidence',
    full_model='NOT_RUN_IN_CI',native=summary(root/'ctest.xml'),mesa=summary(root/'mesa-ctest.xml'),
    capability_skip='FP32 FMA prerequisite tested independently and enforced by application preflight; a skip is not a numerical PASS')
(root/'scope.json').write_text(json.dumps(report,indent=2)+'\n')
PY
