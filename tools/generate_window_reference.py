"""Regenerate window metadata with the pinned official Python source, never C++ output."""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/reference/official_window.py"
EXPECTED_SHA256 = "61b3e0235e25d3e901271186bbaa43a90dcc1552f003daa3d2844b5126ce7e84"
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256, "Reference source changed"
spec = importlib.util.spec_from_file_location("official_seedvr2_window", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
shapes = [(1, 1, 1), (1, 7, 11), (2, 17, 31), (5, 45, 80),
          (9, 45, 80), (31, 13, 27), (3, 68, 120), (1, 25, 576)]
cases = []
for grid in shapes:
    for shifted, function in [(False, module.make_720Pwindows_bysize),
                              (True, module.make_shifted_720Pwindows_bysize)]:
        bounds = [[v for axis in window for v in (axis.start, axis.stop)]
                  for window in function(grid, (4, 3, 3))]
        cases.append({"grid": grid, "shifted": shifted, "windows": bounds})
data = {"source_repository": "ByteDance-Seed/SeedVR",
        "source_commit": "e4de8c24441a67e1b7df56abea10645059bb1185",
        "source_path": "models/dit_v2/window.py", "source_sha256": EXPECTED_SHA256,
        "scope": "Window metadata only; no attention, RoPE or model inference",
        "cases": cases}
target = ROOT / "tests/data/window-reference.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(cases)} independently generated reference cases")
