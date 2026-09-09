# Contributing

Start with [the tutorial](docs/TUTORIAL.md) and [source navigation](docs/ARCHITECTURE.md).
The current target is SeedVR2 **3B**, FP32-B, native Linux, images and bounded short clips.
Please keep changes small enough to explain and validate independently.

1. Reproduce the problem with a small input or one component. Record the commit,
   command, CPU/GPU, driver, backend and exact failure. Attach `run.json` when relevant.
   Inspect reports before sharing: they can contain local filesystem paths.
2. Keep public SDK, CLI/worker, model mathematics, model packaging and resource lifetime
   separate. CLI and Web must call the same native implementation.
3. If mathematics or lowering changes, compare the same inputs against the retained
   official reference. Do not replace golden outputs with candidate outputs, relax old
   tolerances to obtain a pass, or count skipped/empty cases as successes.
4. Run the small native tests and the tests affected by the change. Repeat a passing
   full-model experiment only when the changed implementation or an unresolved question
   warrants its cost. Preserve failures and the identity of every tested binary/SDK.
5. Describe correctness, measured benefit, cost and default behavior for optimizations.
   Do not infer activation/workspace size from total RSS or whole-device GPU memory.

## Local checks

```sh
python3 tools/build_native.py --cli-only --jobs 2
python3 tools/check_source_package.py
bash tools/native_ci.sh build/tutorial dist/tutorial .cache/contributor-ci
```

The last command also requires Mesa Vulkan and validation layers. For frontend changes,
build without `--cli-only`, then run:

```sh
python3 tools/check_web.py dist/tutorial/bin/seedvr2-web dist/tutorial/bin/seedvr2 \
  --output .cache/contributor-web.json
```

Never commit checkpoints, converted full models, local databases, environments, build
directories, tokens or personal input media. Small new fixtures need source attribution
and a clear redistribution license. A new model package must be reviewed before adding
its identity to `policies/reviewed-packages.json`; do not add a bypass to ordinary inference.

Contributions to original project files are accepted under the repository's Apache-2.0
license. Preserve upstream licenses in copied or adapted code. Source publication does
not automatically qualify a binary/model release; see [licensing](docs/LICENSING.md).
