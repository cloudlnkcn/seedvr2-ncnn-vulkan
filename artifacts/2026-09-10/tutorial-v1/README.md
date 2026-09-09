# Tutorial delivery evidence / 教程交付证据

Scope: a fresh native build and small tutorial experiments from source commit
`0837f7cacfe6d8d071cecab939eb82c23e454cd6`. This is not a new full-model acceptance run.
The source tree, host environment, executable and shared SDK hashes are retained.

结论与边界见 [TUTORIAL-READINESS.md](../../../docs/TUTORIAL-READINESS.md)。
本目录不含官方 checkpoint、完整转换模型或可执行二进制。

| Evidence | Meaning |
| --- | --- |
| `source-bindings.json`, `environment.json` | Tested source, native binary/SDK identities and host tools |
| `clean-build-interrupted.log` | First build stopped by the interrupted session; not a passing build |
| `clean-build-resumed.log`, `*-build.log` | Resumed build, 14 CTests, installation and installed version query; dependency logs describe the resumed invocation |
| `installed-documents.json` | Installed license/notice and entry documents match the tested checkout |
| `native-ci/` | 10 CLI checks, 23 engine boundary checks, 15 first-use checks, 11 reference contracts, 14 CTests, 5 Mesa regressions, SDK identity |
| `awa-export.json`, `awa-reference.log`, `awa-export.log` | Four newly generated reference cases and the corresponding pnnx export |
| `awa-fixtures/` | Portable synthetic QKV, official reference outputs and lowered small AWA graphs; all case/suite hashes are unchanged |
| `awa-native.json`, `awa-native.log` | Eight actual CPU/Vulkan comparisons against the four cases; 16 tensor comparisons |
| `official-download-check.json`, `official-*.log` | Real official HTTPS download/resume and final hashes for the two small embedding files |
| `model-download-contract.log` | Nine local HTTP/integrity contract tests, including expected rejection cases |
| `web.json`, `web.log` | 54 local HTTP/CLI/SQLite and CPU AWA checks; full-model jobs excluded |
| `source-package.json` | Local documentation paths, original reference/vendor hashes and tracked inventory at the tested source commit |
| `history-inspection.json` | Selected credential-pattern and >10 MiB blob inspection, not a comprehensive security audit |

The native checkout had no existing build, installed libraries or model files. Only
immutable dependency archives were copied in; Web dependency installation used the
host npm cache. The fixed Python export environment and pnnx executable were reused.
The initial ncnn compilation log was replaced by the preparation helper on resume;
the retained initial top-level log records interruption, and the retained ncnn log
covers the resumed build. No missing output is labeled successful.

`awa-export.json` also records TorchScript/pnnx intermediates that remain in the local
cache. The public `awa-fixtures` directory intentionally includes only files required
by the native cases, their untouched references and pnnx logs. These are synthetic
small operator fixtures, not a converted SeedVR2 checkpoint. A copied fixture's hash
was checked against its recorded case before inclusion.

After building the native application, the fixtures can be replayed with the tutorial's
Python environment (the comparison tool uses NumPy):

```sh
.venv-export/bin/python tools/check_awa.py \
  --binary dist/tutorial/bin/seedvr2 \
  --suite artifacts/2026-09-10/tutorial-v1/awa-fixtures/suite.json \
  --output .cache/tutorial/replay --report .cache/tutorial/replay.json \
  --validation-layer
```

Use a new output directory; use `--backends cpu` if Vulkan is unavailable and report
that reduced scope. The large historical model tensors are not needed for this replay.
GitHub Actions results remain separate from these local measurements.
