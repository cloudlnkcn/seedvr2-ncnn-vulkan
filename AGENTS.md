# SeedVR2 project instructions

- Work only in this standalone project. Do not alter sibling ncnn, pnnx or ERNIE repositories.
- Read docs/DELIVERY-PLAN.md, README.md and the current delivery evidence. Use actual sources/results rather than version labels as evidence.
- Preserve the 3B model mathematics, temporal VAE and clipped adaptive windows. 7B needs a separate implementation review.
- CLI, local Web worker and installed SDK share native computation. Keep Python in conversion/reference tooling.
- A passing diagnostic requires the complete expected tensor contract and a reviewed reference identity. Empty, missing, duplicate or unknown boundaries fail. Never issue model certification from file hashes or a diagnostic PASS.
- Keep historical numerical thresholds and failures. New calibrated protocols need their own version and rationale. Functional completion, numerical errors, perceptual quality and performance are separate.
- Run heavy model jobs sequentially and inspect available memory/GPU first. Snapshot the binary and inputs; do not change them while a run is active.
- Do not repeat passing large experiments unless a relevant change or an unresolved question justifies it.
- Prefer simple responsibility boundaries, small concrete improvements and measured optimizations. No speculative framework or global cache.
- Local source control, construction and testing are authorized. Do not push or publish without user authorization.
