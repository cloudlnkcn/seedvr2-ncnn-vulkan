#pragma once
#include "awa.hpp"
#include "constant.hpp"
#include "video_layers.hpp"
#include "engine_build.hpp"
#include "graph.hpp"
#include "seedvr2/image.hpp"
#include <algorithm>
#include <numbers>
#include <pipelinecache.h>
namespace seedvr2::engine::inference {
using namespace detail;
inline constexpr auto profile = "seedvr2-3b-image-fp32-b-v1";
inline constexpr auto video_profile = "seedvr2-3b-video-fp32-b-v1";
struct Cancelled final : std::exception {
    const char *what() const noexcept override { return "Image processing cancelled"; }
};
struct GraphFiles { std::filesystem::path param, weights; };
struct Package {
    Json manifest;
    std::string identity;
    std::map<std::string, GraphFiles> graphs;
    ncnn::Mat text, time;
    explicit Package(const std::filesystem::path &root, const std::function<void()> &check = {}, bool video = false) {
        const auto path = root/"manifest.json";
        manifest = read_document(path);
        identity = hash(path);
        if (manifest.at("schema_version") != (video ? "seedvr2-video-package-v1" : "seedvr2-image-package-v1") ||
            manifest.at("profile") != (video ? video_profile : profile) || manifest.at("model_id") != "seedvr2-3b" ||
            manifest.at("precision") != "fp32" || manifest.at("ncnn_commit") != SEEDVR2_NCNN_COMMIT ||
            manifest.at("sampling") != Json({{"steps", 1}, {"timestep", 1000}, {"cfg", 1},
                                            {"latent_scale", .9152}, {"color_fix", "none"}}) ||
            !manifest.at("graphs").is_array() || manifest.at("graphs").size() != 36)
            throw std::runtime_error("Unsupported or incomplete image model package");
        std::set<std::string> expected{"encoder", "decoder", "patch-in", "patch-out"};
        for (int i = 0; i < 32; ++i)
            expected.insert("block-"+std::string(i < 10 ? "0" : "")+std::to_string(i));
        for (const auto &row : manifest.at("graphs")) {
            if (check) check();
            const auto id = row.at("id").get<std::string>();
            if (expected.erase(id) != 1)
                throw std::runtime_error("Duplicate or unknown graph in model package");
            graphs.emplace(id, GraphFiles{artifact(root, row.at("param"), 128*1024),
                artifact(root, row.at("weights"), 1024ULL*1024*1024)});
        }
        for (const auto *key : {"text", "time"}) {
            const auto &row = manifest.at("constants").at(key);
            const std::vector<int> shape = std::string_view(key) == "text" ?
                std::vector<int>{58, 2560} : std::vector<int>{2560, 6};
            if (row.at("shape") != shape || row.at("dtype") != "f32le")
                throw std::runtime_error("Invalid fixed conditioning shape");
            auto value = allocate_tensor(shape);
            read_tensor(artifact(root, row, 1024*1024), value);
            (std::string_view(key) == "text" ? text : time) = value;
        }
        if (hash(path) != identity)
            throw std::runtime_error("Model package changed during validation");
    }
};

// Explicit, versioned generator, independent of std::normal_distribution and
// Python. Same-seed equivalence to the PyTorch generator is never claimed.
class NormalNoise {
public:
    explicit NormalNoise(std::uint64_t seed) : state_(seed) {}
    float next() {
        if (has_spare_) { has_spare_ = false; return spare_; }
        const double radius = std::sqrt(-2.*std::log(uniform()));
        const double angle = 2.*std::numbers::pi*uniform();
        spare_ = static_cast<float>(radius*std::sin(angle));
        has_spare_ = true;
        return static_cast<float>(radius*std::cos(angle));
    }
private:
    double uniform() {
        auto z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z^(z>>30))*0xbf58476d1ce4e5b9ULL;
        z = (z^(z>>27))*0x94d049bb133111ebULL;
        z ^= z>>31;
        return (static_cast<double>(z>>11)+.5)/9007199254740992.;
    }
    std::uint64_t state_;
    float spare_ = 0;
    bool has_spare_ = false;
};

inline std::vector<ncnn::Mat> execute_graph(const GraphFiles &files, const std::string &id,
    const std::vector<ncnn::Mat> &inputs, const ImageRequest &request, int gpu,
    Json &report, const std::function<void()> &check, ncnn::PipelineCache *pipelines) {
    const auto started = Clock::now();
    ExecutionTrace awa;
    ncnn::Net net;
    net.opt.use_vulkan_compute = request.vulkan;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = net.opt.use_bf16_packed = false;
    net.opt.use_winograd_convolution = net.opt.use_cooperative_matrix = false;
    net.opt.num_threads = request.threads;
    net.opt.pipeline_cache = pipelines;
    if (request.vulkan) net.set_vulkan_device(gpu);
    if (register_awa(net, awa) != 0 || register_constant(net) != 0 || register_video_layers(net) != 0 ||
        net.load_param(files.param.string().c_str()) != 0 || net.layers().size() > 512 ||
        net.input_indexes().size() != inputs.size())
        throw std::runtime_error("Cannot load graph structure: "+id);
    const bool dit = id.starts_with("block-");
    if (net.output_indexes().size() != (dit ? 2 : 1))
        throw std::runtime_error("Graph output arity mismatch: "+id);
    const std::set<std::string> allowed{"Input", "Convolution", "GroupNorm", "Swish", "BinaryOp",
        "Permute", "PixelShuffle", "Reshape", "SDPA", "Split", "SeedVR2Constant", "InnerProduct",
        "RMSNorm", "Slice", "SeedVR2AWA", "Gemm",
        "SeedVR2TemporalConv", "SeedVR2FrameNorm", "SeedVR2TemporalShuffle", "SeedVR2FrameSDPA"};
    unsigned awa_count = 0;
    for (const auto *layer : net.layers()) {
        if (!allowed.contains(layer->type) || (request.vulkan && layer->type != "Input" && !layer->support_vulkan))
            throw std::runtime_error("Unsupported "+id+" layer: "+layer->type);
        awa_count += layer->type == "SeedVR2AWA";
    }
    if (awa_count != (dit ? 1U : 0U) || (dit &&
        (awa.heads != 20 || awa.shifted != std::stoi(id.substr(6))%2)))
        throw std::runtime_error("Adaptive attention graph metadata mismatch");
    check();
    if (net.load_model(files.weights.string().c_str()) != 0)
        throw std::runtime_error("Cannot load graph weights: "+id);
    const auto loaded = Clock::now();
    check();
    std::vector<ncnn::Mat> outputs(net.output_indexes().size());
    Json layers = Json::array();
    if (request.vulkan) {
        const auto *device = net.vulkan_device();
        VulkanAllocators allocators(device);
        auto options = net.opt;
        options.blob_vkallocator = options.workspace_vkallocator = allocators.blob;
        options.staging_vkallocator = allocators.staging;
        ncnn::VkCompute command(device);
        std::vector<ncnn::VkMat> uploaded(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i)
            command.record_upload(inputs[i], uploaded[i], options);
        auto results = graph_forward(net, uploaded, &command, options, layers, check, (id == "encoder" || id == "decoder") ? 4 : 8);
        for (std::size_t i = 0; i < results.size(); ++i) {
            auto result = results[i];
            if (result.elempack != 1) {
                ncnn::VkMat unpacked;
                device->convert_packing(result, unpacked, 1, command, options);
                result = unpacked;
            }
            command.record_download(result, outputs[i], options);
        }
        if (command.submit_and_wait() != 0)
            throw std::runtime_error("Vulkan image graph submission failed: "+id);
    } else {
        outputs = graph_forward(net, inputs, nullptr, net.opt, layers, check);
    }
    if (dit && (request.vulkan ? (awa.vulkan_calls != 1 || awa.cpu_calls != 0) :
                                (awa.cpu_calls != 1 || awa.vulkan_calls != 0)))
        throw std::runtime_error("Requested attention backend did not execute exclusively");
    const auto end = Clock::now();
    report.push_back({{"id", id}, {"backend", request.vulkan ? "ncnn-vulkan" : "ncnn-cpu"},
        {"load_ms", std::chrono::duration<double, std::milli>(loaded-started).count()},
        {"compute_ms", std::chrono::duration<double, std::milli>(end-loaded).count()},
        {"layers", layers.size()}, {"cpu_layers", request.vulkan ? 0 : layers.size()},
        {"vulkan_layers", request.vulkan ? layers.size() : 0},
        {"awa_windows", awa.windows.load()}, {"awa_cpu", awa.cpu_calls.load()}, {"awa_vulkan", awa.vulkan_calls.load()}});
    return outputs;
}
inline void expect(const ncnn::Mat &value, const std::vector<int> &shape, const std::string &stage) {
    if (tensor_shape(value) != shape || value.elempack != 1 || value.elemsize != 4)
        throw std::runtime_error("Unexpected tensor shape at "+stage);
}

}
