# Source and dependency licenses

Original project code and documentation are licensed under [Apache-2.0](../LICENSE).
Attribution and exceptions are listed in [NOTICE](../NOTICE). This is an independent
educational port of SeedVR2 3B, not an official ByteDance or Tencent distribution.

| Material | License / provenance |
| --- | --- |
| Original C++ application, SDK, shaders, tooling and documentation | Apache-2.0 |
| Retained SeedVR reference files and model-derived implementations | Apache-2.0; original files and hashes in [tests/reference/seedvr](../tests/reference/seedvr) |
| ncnn and pnnx | BSD-3-Clause; [original license](../third_party/engine/ncnn-LICENSE.txt) and separately pinned runtime/converter archives |
| nlohmann/json and historical cpp-httplib header | MIT; notices beside the vendored headers |
| CLI11, Drogon, trantor and JsonCpp | Original licenses copied during installation from their locked source archives |
| Frontend dependencies | [Production dependency notices](../third_party/studio/NOTICES.txt) and [additional sources](../third_party/studio/license-sources.json) |
| glslang, SPIRV-Tools and stb | Original texts in [third_party/engine](../third_party/engine) |
| JPEG and FFmpeg libraries | System dependencies; recorded licenses/configuration in [third_party/media](../third_party/media) |
| NASA astronaut fixture | Public domain according to scikit-image; [source and transformations](../tests/fixtures/natural/provenance.json) |
| Official model checkpoints | Downloaded separately from the [pinned official repository](https://huggingface.co/ByteDance-Seed/SeedVR2-3B/tree/37255ff8cccfb01071b87f635a5948ca8d53117c); not included in Git |

The Apache license requires retaining the applicable license and attribution notices;
see the [Apache license text](https://www.apache.org/licenses/LICENSE-2.0).
Copied upstream source files remain unchanged so their recorded hashes still apply.
Project adaptations and the allocator overlay are identified in
[provenance.md](provenance.md) and [MEMORY-VALIDATION.md](MEMORY-VALIDATION.md).

## Source publication and binary distribution

The public repository contains source, small fixtures and historical reports. It does
not publish the local `dist/` installation, official checkpoints or converted packages.

The tested media configuration enables GPL components in FFmpeg and uses libx264.
[FFmpeg's own licensing guidance](https://ffmpeg.org/legal.html) explains that enabling
GPL components changes the applicable FFmpeg license. A source checkout's Apache-2.0
license does not make a combined application binary Apache-only or remove the relevant
GPL source/notice obligations. Any future binary distribution must document its actual
library configuration and supply the corresponding materials required by those licenses.
Current build instructions use system libraries; no portable binary release is offered.

The current source does not silently switch codec or disable video to change that scope.
Model file integrity checks authenticate the project's reviewed export payload, not a
new copyright license, official signature, endorsement or quality certificate.
