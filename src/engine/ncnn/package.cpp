#include "package.hpp"
#include "reviewed_packages.hpp"

namespace seedvr2::engine::inference {
PackageManifest inspect_manifest(const std::filesystem::path &root, bool video) {
    const auto path = root/"manifest.json";
    if (std::filesystem::is_symlink(path)) throw std::runtime_error("Model manifest must be a regular local file");
    PackageManifest result{read_document(path), hash(path)};
    const auto reviewed = Json::parse(SEEDVR2_REVIEWED_PACKAGES);
    // Build-machine/exporter metadata may change on a fresh reproducible export.
    // Every model/source identity, graph, constant, bound and sampling field
    // remains in the authenticated payload; the full manifest is also reported.
    auto payload=result.document; payload.erase("exporter");
    const auto payload_identity=hash_text(payload.dump());
    bool known = false;
    for (const auto &row : reviewed.at("packages"))
        known |= row.at("payload_sha256") == payload_identity && row.at("kind") == (video ? "video" : "image");
    if (!known) throw std::runtime_error("Model package identity is not reviewed for this build; use the documented pinned export");
    const auto &manifest = result.document;
    if (manifest.at("schema_version") != (video ? "seedvr2-video-package-v1" : "seedvr2-image-package-v1") ||
        manifest.at("profile") != (video ? video_profile : profile) || manifest.at("model_id") != "seedvr2-3b" ||
        manifest.at("precision") != "fp32" || manifest.at("reference_profile") != "FP32-B" ||
        manifest.at("sampling") != Json({{"steps",1},{"timestep",1000},{"cfg",1},{"latent_scale",.9152},{"color_fix","none"}}) ||
        !manifest.at("graphs").is_array() || manifest.at("graphs").size() != 36)
        throw std::runtime_error("Unsupported or incomplete model package");
    std::set<std::string> expected{"encoder", "decoder", "patch-in", "patch-out"};
    for (int i=0; i<32; ++i) expected.insert("block-"+std::string(i<10?"0":"")+std::to_string(i));
    for (const auto &row : manifest.at("graphs")) {
        if (expected.erase(row.at("id").get<std::string>()) != 1) throw std::runtime_error("Duplicate or unknown model graph");
        for (const auto *key : {"param", "weights"}) {
            const auto file = artifact(root, row.at(key), std::string_view(key)=="param"?128*1024:1024ULL*1024*1024, false);
            result.bytes += std::filesystem::file_size(file);
        }
        result.largest_graph_bytes = std::max(result.largest_graph_bytes, row.at("weights").at("bytes").get<std::uint64_t>());
    }
    for (const auto *key : {"text", "time"}) {
        const auto file = artifact(root, manifest.at("constants").at(key), 1024*1024, false);
        result.bytes += std::filesystem::file_size(file);
    }
    result.bytes += std::filesystem::file_size(path);
    if (hash(path) != result.identity) throw std::runtime_error("Model manifest changed during inspection");
    return result;
}

Package::Package(const std::filesystem::path &root, const std::function<void()> &check, bool video,
                 ProgressObserver observer) {
    const auto started = Clock::now();
    auto metadata = inspect_manifest(root, video);
    manifest = std::move(metadata.document); identity = std::move(metadata.identity);
    for (const auto &row : manifest.at("graphs")) {
        if (check) check();
        if (observer) observer({"validating",0,38,std::chrono::duration<double,std::milli>(Clock::now()-started).count()});
        const auto id = row.at("id").get<std::string>();
        graphs.emplace(id, GraphFiles{artifact(root,row.at("param"),128*1024),
                                     artifact(root,row.at("weights"),1024ULL*1024*1024)});
    }
    for (const auto *key : {"text", "time"}) {
        if (check) check();
        const auto &row = manifest.at("constants").at(key);
        const std::vector<int> shape = std::string_view(key)=="text"?std::vector<int>{58,2560}:std::vector<int>{2560,6};
        if (row.at("shape") != shape || row.at("dtype") != "f32le") throw std::runtime_error("Invalid fixed conditioning");
        auto value = allocate_tensor(shape); read_tensor(artifact(root,row,1024*1024),value);
        (std::string_view(key)=="text"?text:time) = value;
    }
    if (hash(root/"manifest.json") != identity) throw std::runtime_error("Model package changed during validation");
}

RestoreRequest public_request(const ImageRequest &r, bool video, int frames) {
    return {r.model_directory,r.input_file,r.output_directory,video?MediaKind::video:MediaKind::image,
            r.vulkan?Backend::vulkan:Backend::cpu,r.gpu_index,r.threads,r.long_side,frames,r.seed,r.diagnostic_tensors,
            r.mapped_weights?WeightIO::mapped:WeightIO::buffered,r.memory};
}
} // namespace seedvr2::engine::inference
