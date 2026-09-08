"""Repeatable CLI checks. Python is used for development only."""
from pathlib import Path
import argparse
import datetime
import json
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("executable", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
executable = args.executable.resolve()
base = json.loads((root / "examples/plan-720p.json").read_text())
results = []

def check(name, arguments, exit_code, predicate):
    result = subprocess.run([str(executable), *arguments], capture_output=True,
                            text=True, encoding="utf-8", timeout=10)
    if result.returncode != exit_code:
        raise AssertionError((name, result.returncode, result.stdout, result.stderr))
    data = json.loads(result.stdout)
    if not predicate(data):
        raise AssertionError((name, data))
    results.append({"case": name, "status": "PASS", "exit_code": exit_code})

with tempfile.TemporaryDirectory(prefix="seedvr2-cli-") as temporary:
    folder = Path(temporary)
    good = folder / "中文 路径.json"
    good.write_text(json.dumps(base), encoding="utf-8")
    check("unicode-and-space-path", ["plan", "--request", str(good)], 0,
          lambda x: x["video_tokens"] == 18000 and x["runnable"] is False)
    check("missing-request", ["plan", "--request", str(folder / "missing.json")], 3,
          lambda x: x["error"]["code"] == "FILE_READ")
    check("run-requires-real-input-and-model", ["run"], 2,
          lambda x: x["error"]["code"] == "CLI_USAGE")
    check("unknown-argument", ["plan", "--typo", str(good)], 2,
          lambda x: x["error"]["code"] == "CLI_USAGE")
    check("negative-window-axis", ["windows", "--tokens", "-1", "45", "80"], 2,
          lambda x: x["error"]["code"] == "INVALID_DIMENSION")
    check("huge-window-grid", ["windows", "--tokens", *(3 * ["4294967295"])], 2,
          lambda x: x["error"]["code"] == "PLANNING_LIMIT")
    check("shifted-metadata", ["windows", "--tokens", "5", "45", "80", "--shifted"], 0,
          lambda x: len(x["windows"]) == 48 and x["attention_execution_status"] == "NOT_RUN")
    bad = folder / "invalid.json"
    bad.write_text('{"schema_version":"1.0","schema_version":"2.0"}', encoding="utf-8")
    check("duplicate-key", ["plan", "--request", str(bad)], 2,
          lambda x: x["error"]["code"] == "DUPLICATE_KEY")
    bad.write_text('{"schema_version":"0.2-design","design_example":true}', encoding="utf-8")
    check("legacy-design-job-rejected", ["plan", "--request", str(bad)], 2,
          lambda x: x["error"]["code"] == "SCHEMA_UNSUPPORTED")
    bad.write_bytes(b" " * (1024 * 1024 + 1))
    check("oversized-file", ["plan", "--request", str(bad)], 3,
          lambda x: x["error"]["code"] == "REQUEST_TOO_LARGE")

report = {"checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "scope": "CLI command boundaries; numerical inference has separate complete-image evidence", "cases": results}
text = json.dumps(report, indent=2) + "\n"
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
print(text, end="")
