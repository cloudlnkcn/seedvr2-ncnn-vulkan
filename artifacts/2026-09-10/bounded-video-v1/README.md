# Bounded natural-video validation — 2026-09-10

**Execution completed; two numerical cases failed and descriptive quality results were negative.**
This archive covers one frozen SeedVR2 3B single-step FP32-B installation on Linux
RTX 4060 Laptop Vulkan, three 128×80 development clips, and four additional AWA cases.
The full [results and reproduction commands](../../../docs/BOUNDED-VIDEO-RESULTS.md)
explain the scope and unresolved numerical sensitivity.

| Case | Full tensor boundaries | Native / official RGB8 maximum difference |
| --- | ---: | ---: |
| motion-9 | 63/73, failed | 1 |
| padding-8 | 71/73, failed | 1 |
| cut-17 | 73/73 | 1 |

All three native and official FP32-B results have lower fixed-target PSNR/SSIM than
the bicubic baseline. These clips share one source and artificial degradation;
they are not a representative benchmark. No quality threshold was selected after
seeing the output. Model certification remains false.

- [summary.json](summary.json): separate execution, numerical, quality, resource and isolation results.
- [protocol.md](protocol.md) and [plan.json](plan.json): declaration written before inference, frozen inputs/scripts/binary identities; the original plan's reference review remains `PENDING` as historical data.
- [audit.json](audit.json): independent recomputation of 73 boundaries for each case, 54,208,000 compared scalars per side; source/tensor audit success does not mean numerical success.
- [audit-final.json](audit-final.json): the same retained tensors checked after accepting both `lib`/`lib64` SDK layouts; original audit/script bytes remain retained.
- `*-run.json`, `*-reference.json`, `execution/`: original reports, measurements, stderr/stdout and official CPU resource records. Official checker exit 1 means the retained numerical failure after completed reference generation.
- `*-output.mp4`: actual native application outputs, bound by the corresponding run reports.
- `quality/<case>/`: per-frame metrics, comparison images and separately re-encoded bicubic/native/official/target diagnostic previews.
- [awa-results.json](awa-results.json), `awa/`: four pnnx exports, 8/8 CPU/Vulkan cases and 16 output comparisons; maximum absolute error `1.6093254e-6`.
- `isolation/`: native replay of the failure, same-input component checks, and the 32-block official-start trajectory. The failing block-19 replay reproduces both native output tensors byte for byte.
- `logs/`: original numerical failures, the square-grid and SDK-loader diagnostic failures, and the empty-input metric regression before/after repair.
- [source-bindings.json](source-bindings.json): all 108 frozen file identities; matching source files resolve against the repository, exact historical overrides are retained. CLI/SDK binaries and raw full trajectories stay local.
- [native-snapshot-smoke.json](native-snapshot-smoke.json): exact loaded CLI/SDK identity after snapshot copying was narrowed to SeedVR2 files; no model inference was repeated for this tooling change.
- [inventory.json](inventory.json): hashes of archived evidence and committed small fixture payloads; excludes itself.

The 3 new reference reports were admitted only after source binding, raw-tensor
recomputation and frame review. Their numerical failures remain unchanged. Registry
admission supports strict replay; it is not model quality certification.

Same-input motion blocks 19–31 pass 26/26 outputs; padding blocks 15–17 pass 6/6.
Starting the full motion DiT trajectory from official patch/text/time inputs passes
64/64 outputs. This supports amplification of differences entering DiT, without
isolating one upstream operator or fixing the original 63/73 and 71/73 trajectories.
No production model mathematics, weights, thresholds or default options changed.

The fixed inputs/targets are committed under
[tests/fixtures/video-bounded](../../../tests/fixtures/video-bounded/SOURCE.md),
with ImageIO provenance and the original manifest. Historical absolute paths in
reports refer to the measurement host. Large raw tensors, checkpoints and compiled
models are not in Git. Diagnostic previews are lossy viewing aids; numerical metrics
were calculated before video encoding. Only frame-level visual inspection was
performed, not temporal blind review.

Single-run wall time and RSS are descriptive; whole-card GPU memory includes desktop
use. They do not establish speedup or allocation-class peaks. Remote CI for this
change has not run; historical native CI has separately recorded commit identities.
