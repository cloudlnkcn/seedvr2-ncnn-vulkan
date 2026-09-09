# GitHub native CI evidence / 远端原生验证

The final checked CI commit is `b8e6611788272e25d9e4c0f463ba0a7f36a343b2`.
All three native Ubuntu 24.04 jobs passed. This directory retains downloaded
GitHub artifacts, the complete workflow output, API run metadata and file hashes.
The evidence/documentation commit that adds this directory uses `[skip ci]` to
avoid repeating the unchanged build and numerical tests for an archival change.

- [First run: 34409538349](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/actions/runs/34409538349), source `0a0ff9ae098200c3070951bb6cd99ed55df4f1f6`.
- [Final run: 34410376931](https://github.com/mingshi2333/seedvr2-ncnn-vulkan/actions/runs/34410376931), source `b8e6611788272e25d9e4c0f463ba0a7f36a343b2`.
- [Machine-readable summary](summary.json), [first-run file hashes](first-files.json), [final-run file hashes](final-files.json).

| Final job | CTest | Mesa Vulkan regressions | Web checks | Job time |
| --- | --- | --- | --- | --- |
| GCC CLI | 14/14 | 5/5 | Not built by this job | 442 s |
| Clang CLI | 14/14 | 5/5 | Not built by this job | 462 s |
| GCC Web | 14/14 | 5/5 | 54/54 | 691 s |

The 5 Vulkan regressions are part of the native test set, exercised explicitly under
Mesa again; do not add them to 14 as if there were 19 unique tests. There were no
skipped native/Mesa test cases. Each job also checked 10 CLI boundaries, 23 engine
boundaries, 15 first-use cases, 11 reference contracts, actual installed SDK identity,
and the source/downloader contracts. Full 3B weights were not downloaded or executed.
These job times include setup, downloads and compilation, not model inference.

## Retained failure of evidence collection

The first run passed all three jobs, but its artifact upload omitted dependency logs
inside `.cache`. The pinned upload action ignores hidden directories by default;
see the [action's own documented behavior](https://github.com/actions/upload-artifact/blob/ea165f8d65b6e75b540449e92b4886f43607fa02/README.md#uploading-hidden-files).
The first run's other reports and complete workflow output remain in `first/`.
Its missing dependency logs are not reconstructed or labeled present.

The follow-up changed only `.github/workflows/native.yml`: opt in to hidden files
while naming four exact dependency log paths, rather than uploading the cache tree.
The second run passed and its downloaded archives actually contained all 8 expected
dependency logs across the three jobs: ncnn/CLI11 for both CLI builds, plus
ncnn/CLI11/JsonCpp/Drogon for the Web build. Presence, nonempty content and SHA-256
are recorded in `summary.json` and `final-files.json`.

The raw workflow output retains the runner's Node.js 20 deprecation warning for the
pinned upload action and its forced Node.js 24 execution. Both uploads completed;
that warning is not reported as a failed test or hidden from the record.

## Scope and identities

`first/` and `final/` keep the original archive layout. Their `.cache` folders contain
only the downloaded dependency build logs; they contain no checkpoints or binaries.
`run.json` identifies the source commit and individual jobs; `artifacts.json` retains
GitHub's artifact metadata. SDK and CLI hashes are in each job's `artifacts/ci` reports.

The native runtime, exporter and reference sources are unchanged from the locally
tested tutorial source `0837f7cacfe6d8d071cecab939eb82c23e454cd6`; the intervening
changes are entry documents, retained evidence and the upload configuration fix.
[Local tutorial evidence](../tutorial-v1/README.md) records physical-GPU small-operator
comparisons. [The earlier 0.7.0 evidence](../../2026-09-08/memory-v1/README.md) records
full-model checks against its own binary, inputs and device. Ubuntu software Vulkan
CI does not replace those checks or certify image/video quality.
