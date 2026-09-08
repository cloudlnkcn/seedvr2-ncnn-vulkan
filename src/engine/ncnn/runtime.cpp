#include "awa.hpp"
#include "video_layers.hpp"
#include "graph.hpp"
#include "constant.hpp"
#include "engine_build.hpp"
#include "fixture.hpp"
#include "seedvr2/engine.hpp"
#include "seedvr2/path.hpp"
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <command.h>
#include <fstream>
#include <gpu.h>
#include <iomanip>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <sstream>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace seedvr2::engine {
namespace {
using namespace detail;
} // namespace
std::string build_status() {
    return Json{{"ncnn_linked", true},
                {"ncnn_commit", SEEDVR2_NCNN_COMMIT},
                {"ncnn_version", NCNN_VERSION_STRING},
                {"awa_cpu", true},
                {"awa_vulkan", true},
                {"bundled_awa_self_test", true},
                {"vae_image_diagnostics", true},
                {"dit_block_diagnostics", true},
                {"pnnx_submodel_exports", true},
                {"reference_profile", "FP32-B"},
                {"precision", "fp32"},
                {"model_inference", true},
                {"single_image_restoration", true},
                {"video_restoration", true},
                {"vae_video_diagnostics", true},
                {"max_video_frames",17},
                {"max_video_side",128},
                {"video_streaming_cache",false},
                {"model_verified", false}}
        .dump();
}
Result<std::string> devices() {
    try {
        std::lock_guard lock(engine_mutex);
        init_gpu();
        Json list = Json::array();
        for (int i = 0; i < ncnn::get_gpu_count(); ++i)
            list.push_back(device_info(i));
        return Json{{"schema_version", "1.0"},
                    {"devices", list},
                    {"default_gpu", ncnn::get_default_gpu_index()},
                    {"probe_only", true},
                    {"model_verified", false}}
            .dump(2);
    } catch (const std::exception &e) {
        return Error{"DEVICE_PROBE_FAILED", "engine", e.what()};
    }
}
Result<std::string> run_awa(const CaseRequest &request) {
    try {
        if constexpr (std::endian::native != std::endian::little)
            throw std::runtime_error("f32le input requires a little endian host");
        std::lock_guard lock(engine_mutex);
        const auto started = Clock::now();
        if (request.threads < 1 || request.threads > 32)
            throw std::runtime_error("threads must be 1..32");
        if (request.gpu_index < -1)
            throw std::runtime_error("gpu must be -1 (default) or a nonnegative index");
        const auto case_hash = hash(request.case_file);
        const auto doc = read_case(request.case_file);
        const auto grid = dimensions(doc.at("grid"), {256, 128, 128});
        const int heads = integer(doc.at("heads"), 1, 20),
                  text_length = integer(doc.at("text_length"), 1, 256);
        if (!doc.at("shifted").is_boolean())
            throw std::runtime_error("AWA shifted must be a boolean");
        const auto validate_input = [](const Json &row, const std::vector<int> &shape) {
            if (row.at("dtype") != "f32le" || !row.at("shape").is_array() || row.at("shape").size() != shape.size())
                throw std::runtime_error("AWA input metadata mismatch");
            for (std::size_t i = 0; i < shape.size(); ++i)
                integer(row.at("shape").at(i), shape[i], shape[i]);
        };
        validate_input(doc.at("video_qkv"), {grid[0], grid[1], grid[2], 3 * heads * 128});
        validate_input(doc.at("text_qkv"), {text_length, 3 * heads * 128});
        if (heads < 1 || heads > 20 || text_length < 1 || text_length > 256 || grid[0] < 1 ||
            grid[0] > 256 || grid[1] < 1 || grid[1] > 128 || grid[2] < 1 || grid[2] > 128)
            throw std::runtime_error("AWA case dimensions exceed operator bounds");
        const std::uint64_t input_bytes =
            static_cast<std::uint64_t>(grid[0]) * grid[1] * grid[2] * 3 * heads * 128 * 4;
        if (input_bytes > 128 * 1024 * 1024)
            throw std::runtime_error("AWA input exceeds 128 MiB");
        const auto root = request.case_file.parent_path();
        const auto params = artifact(root, doc.at("model_param"));
        const auto weights = artifact(root, doc.at("model_bin"));
        const auto video = artifact(root, doc.at("video_qkv"));
        const auto text = artifact(root, doc.at("text_qkv"));
        if (std::filesystem::exists(request.output_directory) &&
            !std::filesystem::is_empty(request.output_directory))
            throw std::runtime_error("Output directory is not empty");
        ExecutionTrace trace;
        ncnn::Net net;
        net.opt.use_vulkan_compute = request.vulkan;
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_storage = false;
        net.opt.use_bf16_packed = false;
        net.opt.use_packing_layout = false;
        net.opt.lightmode = false;
        net.opt.num_threads = request.threads;
        Json gpu = nullptr;
        if (request.vulkan) {
            init_gpu();
            const int index =
                request.gpu_index < 0 ? ncnn::get_default_gpu_index() : request.gpu_index;
            if (index < 0 || index >= ncnn::get_gpu_count())
                throw std::runtime_error("Requested Vulkan device is unavailable");
            gpu = device_info(index);
            net.set_vulkan_device(index);
        }
        if (register_awa(net, trace) != 0 || net.load_param(params.string().c_str()) != 0 ||
            net.layers().size() != 3 || net.input_indexes().size() != 2 || net.output_indexes().size() != 2 ||
            trace.heads != heads || trace.shifted != int(doc.at("shifted").get<bool>()))
            throw std::runtime_error("AWA graph and case configuration disagree");
        for (const auto *layer : net.layers())
            if (layer->type != "Input" && layer->type != "SeedVR2AWA")
                throw std::runtime_error("Unexpected AWA diagnostic layer");
        if (net.load_model(weights.string().c_str()) != 0)
            throw std::runtime_error("Cannot load the AWA ncnn graph");
        ncnn::Mat vin(3 * heads * 128, grid[2], grid[1], grid[0], size_t(4), 1);
        ncnn::Mat tin(3 * heads * 128, text_length, size_t(4), 1);
        if (vin.empty() || tin.empty())
            throw std::runtime_error("Input allocation failed");
        read_tensor(video, vin);
        read_tensor(text, tin);
        auto extractor = net.create_extractor();
        if (extractor.input("in0", vin) != 0 || extractor.input("in1", tin) != 0)
            throw std::runtime_error("AWA input binding failed");
        const auto compute_start = Clock::now();
        ncnn::Mat vout, tout;
        if (extractor.extract("out0", vout) != 0 || extractor.extract("out1", tout) != 0)
            throw std::runtime_error("AWA execution failed (shape, scratch budget, or backend)");
        const auto compute_end = Clock::now();
        if (request.vulkan ? (trace.vulkan_calls != 1 || trace.cpu_calls != 0)
                           : (trace.cpu_calls != 1 || trace.vulkan_calls != 0))
            throw std::runtime_error("Requested AWA backend was not exclusively executed");
        if (vout.dims != 4 || vout.w != heads * 128 || vout.h != grid[2] || vout.d != grid[1] ||
            vout.c != grid[0] || tout.dims != 2 || tout.w != heads * 128 || tout.h != text_length)
            throw std::runtime_error("AWA output shape mismatch");
        // Detect accidental changes while the graph and inputs were consumed.
        artifact(root, doc.at("model_param"));
        artifact(root, doc.at("model_bin"));
        artifact(root, doc.at("video_qkv"));
        artifact(root, doc.at("text_qkv"));
        if (hash(request.case_file) != case_hash)
            throw std::runtime_error("AWA case changed during execution");
        std::filesystem::create_directories(request.output_directory);
        Json outputs = {{"video", write_tensor(request.output_directory, "video.f32", vout)},
                        {"text", write_tensor(request.output_directory, "text.f32", tout)}};
        outputs["video"]["shape"] = {grid[0], grid[1], grid[2], heads * 128};
        outputs["text"]["shape"] = {text_length, heads * 128};
        Json report = {
            {"schema_version", "awa-execution-v1"},
            {"status", "EXECUTED"},
            {"case_id", doc.at("case_id")},
            {"case_sha256", case_hash},
            {"backend", request.vulkan ? "ncnn-vulkan" : "ncnn-cpu"},
            {"precision", "fp32"},
            {"ncnn_commit", SEEDVR2_NCNN_COMMIT},
            {"device", gpu},
            {"cpu_calls", trace.cpu_calls.load()},
            {"vulkan_calls", trace.vulkan_calls.load()},
            {"sdpa_calls", trace.sdpa_calls.load()},
            {"windows", trace.windows.load()},
            {"outputs", outputs},
            {"model_verified", false},
            {"numerical_validation", "NOT_PERFORMED_BY_RUNNER"},
            {"timing_ms",
             {{"extract_including_transfers",
               std::chrono::duration<double, std::milli>(compute_end - compute_start).count()},
              {"total",
               std::chrono::duration<double, std::milli>(Clock::now() - started).count()}}}};
        const auto result = report.dump(2);
        std::ofstream file(request.output_directory / "run.json");
        file << result << '\n';
        file.close();
        if (!file)
            throw std::runtime_error("Cannot write run report");
        return result;
    } catch (const std::exception &e) {
        return Error{"AWA_EXECUTION_FAILED", "engine", e.what()};
    }
}
namespace {
Result<std::string> run_graph_case(const CaseRequest &request, bool dit) {
    try {
        if constexpr (std::endian::native != std::endian::little)
            throw std::runtime_error("f32le input requires a little endian host");
        std::lock_guard lock(engine_mutex);
        const auto started = Clock::now();
        if (request.threads < 1 || request.threads > 32)
            throw std::runtime_error("threads must be 1..32");
        if (request.gpu_index < -1)
            throw std::runtime_error("gpu must be -1 (default) or a nonnegative index");
        const auto case_hash = hash(request.case_file);
        const auto doc = read_document(request.case_file);
        const auto component = doc.at("component").get<std::string>();
        const bool video_vae = component == "vae-video-encoder" || component == "vae-video-decoder";
        if (doc.at("schema_version") != (dit ? "dit-block-case-v1" : "ncnn-graph-case-v1") ||
            doc.at("precision") != "fp32" || doc.at("reference_profile") != "FP32-B" ||
            (dit ? component != "dit-block" :
                   (component != "vae-image-encoder" && component != "vae-image-decoder" && !video_vae)))
            throw std::runtime_error("Unsupported graph case or component");
        Json input_specs = dit ? doc.at("inputs") : Json::array({doc.at("input")});
        std::vector<std::vector<int>> shapes, expected_shapes;
        if (dit) {
            integer(doc.at("block_index"), 0, 31);
            if (!input_specs.is_array() || input_specs.size() != 3)
                throw std::runtime_error("DiT block requires video, text and timestep embedding");
            // A 17-frame clip has five temporal latents. Keep the existing
            // total-token bound while permitting its actual component inputs.
            const std::vector<std::vector<int>> bounds = {{5, 16, 16, 2560}, {256, 2560}, {2560, 6}};
            for (std::size_t i = 0; i < bounds.size(); ++i) {
                const auto &value = input_specs.at(i).at("shape");
                if (!value.is_array() || value.size() != bounds[i].size())
                    throw std::runtime_error("DiT block input rank mismatch");
                std::vector<int> shape;
                for (std::size_t j = 0; j < value.size(); ++j)
                    shape.push_back(integer(value.at(j), 1, bounds[i][j]));
                shapes.push_back(shape);
            }
            if (shapes[0][3] != 2560 || shapes[1][1] != 2560 ||
                shapes[2] != std::vector<int>{2560, 6} ||
                shapes[0][0] * shapes[0][1] * shapes[0][2] > 512)
                throw std::runtime_error("DiT diagnostic input outside supported bounds");
            expected_shapes = {shapes[0], shapes[1]};
        } else if (video_vae) {
            const auto &value = input_specs[0].at("shape");
            if (!value.is_array() || value.size()!=4) throw std::runtime_error("Video VAE expects C,T,H,W");
            const bool encoder=component=="vae-video-encoder";
            std::vector<int> shape={integer(value[0],1,32),integer(value[1],1,encoder?17:5),
                integer(value[2],1,encoder?128:16),integer(value[3],1,encoder?128:16)};
            if (shape[0]!=(encoder?3:16) || (encoder && (shape[2]%8 || shape[3]%8 || (shape[1]-1)%4)))
                throw std::runtime_error("Video VAE dimensions outside diagnostic bounds");
            shapes.push_back(shape);
            expected_shapes.push_back({encoder?32:3,encoder?(shape[1]-1)/4+1:(shape[1]-1)*4+1,
                encoder?shape[2]/8:shape[2]*8,encoder?shape[3]/8:shape[3]*8});
        } else {
            const auto shape = dimensions(input_specs[0].at("shape"), {32, 128, 128});
            const bool encoder = component == "vae-image-encoder";
            if ((encoder && (shape[0] != 3 || shape[1] % 8 || shape[2] % 8)) ||
                (!encoder && (shape[0] != 16 || shape[1] > 16 || shape[2] > 16)))
                throw std::runtime_error("VAE diagnostic input outside single-frame bounds");
            shapes.push_back({shape[0], shape[1], shape[2]});
            expected_shapes.push_back({encoder ? 32 : 3, encoder ? shape[1]/8 : shape[1]*8,
                                      encoder ? shape[2]/8 : shape[2]*8});
        }
        const auto root = request.case_file.parent_path();
        const std::uint64_t weight_limit = (dit || video_vae ? 1024ULL : 256ULL) * 1024 * 1024;
        const auto param = artifact(root, doc.at("model_param"));
        const auto binary = artifact(root, doc.at("model_bin"), weight_limit);
        std::vector<ncnn::Mat> inputs;
        for (std::size_t i = 0; i < shapes.size(); ++i) {
            if (input_specs[i].at("dtype") != "f32le")
                throw std::runtime_error("Graph inputs must be f32le");
            const auto tensor = artifact(root, input_specs[i]);
            auto input = allocate_tensor(shapes[i]);
            if (input.empty())
                throw std::runtime_error("Graph input allocation failed");
            read_tensor(tensor, input);
            inputs.push_back(input);
        }
        if (std::filesystem::exists(request.output_directory) &&
            !std::filesystem::is_empty(request.output_directory))
            throw std::runtime_error("Output directory is not empty");
        ExecutionTrace awa_trace;
        ncnn::Net net;
        net.opt.use_vulkan_compute = request.vulkan;
        net.opt.use_packing_layout = false;
        net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_storage = net.opt.use_bf16_packed = false;
        net.opt.use_winograd_convolution = false;
        net.opt.use_cooperative_matrix = false;
        net.opt.num_threads = request.threads;
        Json gpu = nullptr;
        if (request.vulkan) {
            init_gpu();
            const auto index = request.gpu_index < 0 ? ncnn::get_default_gpu_index() : request.gpu_index;
            if (index < 0 || index >= ncnn::get_gpu_count())
                throw std::runtime_error("Requested Vulkan device is unavailable");
            gpu = device_info(index);
            net.set_vulkan_device(index);
        }
        if (dit && (register_awa(net, awa_trace) != 0 || register_constant(net) != 0))
            throw std::runtime_error("Cannot register AWA");
        if (video_vae && register_video_layers(net)!=0) throw std::runtime_error("Cannot register temporal VAE operators");
        if (net.load_param(param.string().c_str()) != 0 || net.layers().size() > 512 ||
            net.input_indexes().size() != shapes.size() || net.output_indexes().size() != expected_shapes.size())
            throw std::runtime_error("Diagnostic graph arity or structure mismatch");
        const std::set<std::string> allowed_vae = {"Input", "Convolution", "GroupNorm", "Swish",
            "BinaryOp", "Permute", "PixelShuffle", "Reshape", "SDPA", "Split",
            "SeedVR2TemporalConv", "SeedVR2FrameNorm", "SeedVR2TemporalShuffle", "SeedVR2FrameSDPA"};
        const std::set<std::string> allowed_dit = {"Input", "SeedVR2Constant", "InnerProduct", "RMSNorm",
            "Swish", "BinaryOp", "Reshape", "Slice", "Split", "SeedVR2AWA"};
        const auto &allowed = dit ? allowed_dit : allowed_vae;
        unsigned awa_count = 0;
        for (const auto *layer : net.layers()) {
            if (!allowed.contains(layer->type) ||
                (request.vulkan && layer->type != "Input" && !layer->support_vulkan))
                throw std::runtime_error("Unsupported graph layer: " + layer->type);
            awa_count += layer->type == "SeedVR2AWA";
        }
        if (dit && (awa_count != 1 || awa_trace.heads != 20 ||
                    awa_trace.shifted != doc.at("block_index").get<int>() % 2))
            throw std::runtime_error("A complete DiT block must contain exactly one AWA");
        if (net.load_model(binary.string().c_str()) != 0)
            throw std::runtime_error("Cannot load graph weights");
        std::vector<ncnn::Mat> outputs(expected_shapes.size());
        Json trace = Json::array();
        const auto compute_start = Clock::now();
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
            auto results = graph_forward(net, uploaded, &command, options, trace, {}, video_vae ? 4 : 0);
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
                throw std::runtime_error("Vulkan graph submission failed");
        } else {
            outputs = graph_forward(net, inputs, nullptr, net.opt, trace);
        }
        const auto compute_end = Clock::now();
        for (std::size_t i = 0; i < outputs.size(); ++i)
            if (tensor_shape(outputs[i]) != expected_shapes[i])
                throw std::runtime_error("Graph output shape mismatch at out" + std::to_string(i));
        if (dit && (request.vulkan ? (awa_trace.vulkan_calls != 1 || awa_trace.cpu_calls != 0)
                                  : (awa_trace.cpu_calls != 1 || awa_trace.vulkan_calls != 0)))
            throw std::runtime_error("Requested AWA backend was not exclusively executed");
        artifact(root, doc.at("model_param"));
        artifact(root, doc.at("model_bin"), weight_limit);
        for (const auto &spec : input_specs)
            artifact(root, spec);
        if (hash(request.case_file) != case_hash)
            throw std::runtime_error("Graph case changed during execution");
        std::filesystem::create_directories(request.output_directory);
        Json report = {{"schema_version", "ncnn-graph-execution-v1"}, {"status", "EXECUTED"},
            {"case_id", doc.at("case_id")}, {"case_sha256", case_hash}, {"component", component},
            {"backend", request.vulkan ? "ncnn-vulkan" : "ncnn-cpu"}, {"precision", "fp32"},
            {"ncnn_commit", SEEDVR2_NCNN_COMMIT}, {"device", gpu}, {"layers", trace},
            {"cpu_calls", request.vulkan ? 0 : trace.size()},
            {"vulkan_calls", request.vulkan ? trace.size() : 0},
            {"dispatch", "EXPLICIT_PER_LAYER_NO_BACKEND_FALLBACK"}, {"model_verified", false},
            {"numerical_validation", "NOT_PERFORMED_BY_RUNNER"},
            {"timing_ms", {{"compute_including_transfers",
                std::chrono::duration<double, std::milli>(compute_end-compute_start).count()},
                {"total", std::chrono::duration<double, std::milli>(Clock::now()-started).count()}}}};
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            const auto name = dit ? (i == 0 ? "video.f32" : "text.f32") : "output.f32";
            auto out = write_tensor(request.output_directory, name, outputs[i]);
            out["shape"] = expected_shapes[i];
            if (dit)
                report["outputs"][i == 0 ? "video" : "text"] = out;
            else
                report["output"] = out;
        }
        if (dit) {
            report["block_index"] = doc.at("block_index");
            report["awa_dispatch"] = {{"cpu_calls", awa_trace.cpu_calls.load()},
                {"vulkan_calls", awa_trace.vulkan_calls.load()}, {"windows", awa_trace.windows.load()},
                {"sdpa_calls", awa_trace.sdpa_calls.load()}};
        }
        auto result = report.dump(2);
        std::ofstream file(request.output_directory / "run.json");
        file << result << '\n';
        file.close();
        if (!file)
            throw std::runtime_error("Cannot write graph report");
        return result;
    } catch (const std::exception &e) {
        return Error{"GRAPH_EXECUTION_FAILED", "engine", e.what()};
    }
}
} // namespace
Result<std::string> run_graph(const CaseRequest &request) { return run_graph_case(request, false); }
Result<std::string> run_dit_block(const CaseRequest &request) { return run_graph_case(request, true); }
Result<std::string> self_test(bool vulkan, int gpu_index, int threads) {
    struct TemporaryDirectory {
        std::filesystem::path path;
        ~TemporaryDirectory() {
            if (!path.empty()) {
                std::error_code ignored;
                std::filesystem::remove_all(path, ignored);
            }
        }
    } temporary;
    try {
        if (gpu_index < -1 || threads < 1 || threads > 32)
            throw std::runtime_error("Invalid self-test device or thread count");
        std::array<unsigned char, 16> random{};
        if (RAND_bytes(random.data(), random.size()) != 1)
            throw std::runtime_error("Cannot create private diagnostic workspace");
        std::ostringstream suffix;
        suffix << std::hex << std::setfill('0');
        for (auto value : random)
            suffix << std::setw(2) << static_cast<unsigned>(value);
        const auto directory = std::filesystem::temp_directory_path() / ("seedvr2-self-test-" + suffix.str());
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("Diagnostic workspace already exists");
        temporary.path = directory;
        std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
        for (const auto &file : awa_fixture()) {
            const auto path = directory / std::string(file.path);
            std::filesystem::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            out.write(file.bytes.data(), static_cast<std::streamsize>(file.bytes.size()));
            out.close();
            if (!out)
                throw std::runtime_error("Cannot materialize embedded diagnostic");
        }
        const auto provenance = read_document(directory / "provenance.json");
        Json cases = Json::array();
        bool passed = true;
        for (const auto &item : provenance.at("cases")) {
            const auto path = artifact(directory, item);
            const auto doc = read_case(path);
            const auto output = path.parent_path() / "output";
            const auto execution = run_awa({path, output, vulkan, gpu_index, threads});
            Json row = {{"case_id", doc.at("case_id")}, {"passed", false}};
            if (is_error(execution)) {
                const auto &error = std::get<Error>(execution);
                row["error"] = {{"code", error.code}, {"message", error.message}};
                passed = false;
            } else {
                auto run = Json::parse(std::get<std::string>(execution));
                row["execution"] = run;
                row["comparisons"] = Json::array();
                bool case_passed = true;
                for (const auto *name : {"video", "text"}) {
                    const auto reference = artifact(path.parent_path(), doc.at(std::string("reference_")+name));
                    const auto actual = artifact(output, run.at("outputs").at(name));
                    if (std::filesystem::file_size(reference) != std::filesystem::file_size(actual))
                        throw std::runtime_error("Self-test output length mismatch");
                    std::ifstream a(reference, std::ios::binary), b(actual, std::ios::binary);
                    double maximum = 0, squared = 0;
                    std::uint64_t count = 0, violations = 0;
                    float x = 0, y = 0;
                    while (a.read(reinterpret_cast<char*>(&x), sizeof(x))) {
                        if (!b.read(reinterpret_cast<char*>(&y), sizeof(y)) ||
                            !std::isfinite(x) || !std::isfinite(y))
                            throw std::runtime_error("Self-test output is incomplete or non-finite");
                        const double delta = std::abs(static_cast<double>(x)-y);
                        maximum = std::max(maximum, delta);
                        squared += delta*delta;
                        violations += delta > 1e-5 + 1e-4*std::abs(static_cast<double>(x));
                        ++count;
                    }
                    if (!count || a.bad())
                        throw std::runtime_error("Self-test reference cannot be read");
                    row["comparisons"].push_back({{"tensor", name}, {"elements", count},
                        {"max_abs", maximum}, {"rmse", std::sqrt(squared/static_cast<double>(count))},
                        {"violations", violations}, {"passed", violations == 0},
                        {"reference_sha256", hash(reference)}, {"candidate_sha256", hash(actual)}});
                    case_passed = case_passed && violations == 0;
                }
                row["passed"] = case_passed;
                passed = passed && case_passed;
            }
            cases.push_back(row);
        }
        return Json{{"schema_version", "awa-self-test-v1"}, {"document_type", "operator-test"},
                    {"status", passed ? "PASS" : "FAIL"}, {"passed", passed},
                    {"scope", "Synthetic AWA operator smoke test; no checkpoint or whole model"},
                    {"backend", vulkan ? "ncnn-vulkan" : "ncnn-cpu"},
                    {"ncnn_commit", SEEDVR2_NCNN_COMMIT},
                    {"fixture_provenance_sha256", hash(directory / "provenance.json")},
                    {"reference_profile", "FP32-B"}, {"atol", 1e-5}, {"rtol", 1e-4},
                    {"cases", cases}, {"raw_outputs_retained", false}, {"model_verified", false}}
            .dump(2);
    } catch (const std::exception &e) {
        return Error{"SELF_TEST_FAILED", "engine", e.what()};
    }
}
} // namespace seedvr2::engine
