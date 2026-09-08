#include "inference.hpp"
#include "provenance.hpp"
#include "image_io.hpp"
#include <bit>

namespace seedvr2::engine {
using namespace inference;
using namespace detail;

Result<std::string> inspect_image(const std::filesystem::path &path) {
    try {
        const auto pixels = load_image(path);
        return Json{{"width", pixels.width}, {"height", pixels.height},
                    {"sha256", hash(path)}, {"bytes", std::filesystem::file_size(path)}}.dump();
    } catch (const std::exception &e) { return Error{"IMAGE_INVALID", "input", e.what()}; }
}
Result<std::string> inspect_image_package(const std::filesystem::path &directory) {
    try {
        Package p(directory);
        return Json{{"status", "INTEGRITY_CHECKED"}, {"profile", profile},
            {"manifest_sha256", p.identity}, {"graphs", p.graphs.size()},
            {"limits", p.manifest.at("limits")}, {"model_verified", false}}.dump();
    } catch (const std::exception &e) { return Error{"MODEL_PACKAGE_INVALID", "model", e.what()}; }
}
Result<std::string> run_image(const ImageRequest &request, ImageObserver observer, CancellationCheck cancelled) {
    try {
        const auto started=Clock::now();
        const auto ready=seedvr2::preflight(public_request(request));
        if (is_error(ready)) return std::get<Error>(ready);
        if constexpr (std::endian::native != std::endian::little)
            throw std::runtime_error("The FP32 package requires a little endian host");
        if (request.threads < 1 || request.threads > 32 || request.gpu_index < -1)
            throw std::runtime_error("Invalid execution settings");
        std::lock_guard lock(engine_mutex);
        auto check = [&] { if (cancelled && cancelled()) throw Cancelled(); };
        auto progress = [&](std::string stage, int completed) {
            check();
            if (observer) observer({std::move(stage), completed, 38,
                std::chrono::duration<double, std::milli>(Clock::now()-started).count()});
        };
        progress("validating", 0);
        if (std::filesystem::exists(request.output_directory) && !std::filesystem::is_empty(request.output_directory))
            throw std::runtime_error("Output directory must be empty");
        const auto input_hash = hash(request.input_file);
        const auto image = load_image(request.input_file);
        auto prepared = prepare_image(image, request.long_side);
        const int height = prepared.h, width = prepared.w, lh = height/8, lw = width/8;
        const int gh = height/16, gw = width/16;
        const auto validation_started=Clock::now();
        Package package(request.model_directory, check, false, [&](const Progress &) { progress("validating",0); });
        const auto validation_ms=std::chrono::duration<double,std::milli>(Clock::now()-validation_started).count();
        Json gpu_info = nullptr;
        int gpu = -1;
        std::unique_ptr<ncnn::PipelineCache> pipelines;
        if (request.vulkan) {
            init_gpu();
            gpu = request.gpu_index < 0 ? ncnn::get_default_gpu_index() : request.gpu_index;
            if (gpu < 0 || gpu >= ncnn::get_gpu_count())
                throw std::runtime_error("Requested Vulkan device is unavailable");
            gpu_info = device_info(gpu);
            // Most DiT layers have identical shapes. Keep their compiled Vulkan
            // pipelines while releasing each block's much larger weights.
            pipelines = std::make_unique<ncnn::PipelineCache>(ncnn::get_gpu_device(gpu));
        }
        std::filesystem::create_directories(request.output_directory);
        Json diagnostics = Json::object(), stages = Json::array();
        auto capture = [&](const std::string &name, const ncnn::Mat &value) {
            if (request.diagnostic_tensors) {
                auto row = write_tensor(request.output_directory, (name+".f32").c_str(), value);
                row["shape"] = tensor_shape(value);
                diagnostics[name] = row;
            }
        };
        auto execute = [&](const std::string &name, const std::vector<ncnn::Mat> &inputs) {
            return execute_graph(package.graphs.at(name), name, inputs, request, gpu, stages, check, pipelines.get());
        };
        capture("prepared", prepared);
        save_image(request.output_directory/"comparison-input.png", prepared);
        progress("encoding", 1);
        auto posterior = execute("encoder", {prepared}).at(0);
        expect(posterior, {32, lh, lw}, "encoder");
        capture("posterior", posterior);
        prepared.release();
        progress("sampling", 2);
        ncnn::Mat posterior_noise(lw, lh, 16, size_t(4), 1), noise(lw, lh, 16, size_t(4), 1);
        ncnn::Mat conditioned(lw, lh, 16, size_t(4), 1);
        NormalNoise random(request.seed);
        for (int c = 0; c < 16; ++c) {
            auto p = posterior_noise.channel(c), z = conditioned.channel(c);
            auto mean = posterior.channel(c), logvar = posterior.channel(c+16);
            for (int j = 0; j < lh*lw; ++j) {
                p[j] = random.next();
                z[j] = (mean[j]+std::exp(.5f*std::clamp(logvar[j], -30.f, 20.f))*p[j])*.9152f;
            }
        }
        for (int c = 0; c < 16; ++c) {
            auto n = noise.channel(c);
            for (int j = 0; j < lh*lw; ++j) n[j] = random.next();
        }
        capture("posterior-noise", posterior_noise);
        capture("noise", noise);
        capture("conditioned", conditioned);
        posterior.release(); posterior_noise.release();
        ncnn::Mat patches(132, gh*gw, size_t(4), 1);
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) {
                auto *p = patches.row(y*gw+x);
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int offset = (dy*2+dx)*33;
                        for (int c = 0; c < 16; ++c) {
                            p[offset+c] = noise.channel(c).row(2*y+dy)[2*x+dx];
                            p[offset+c+16] = conditioned.channel(c).row(2*y+dy)[2*x+dx];
                        }
                        p[offset+32] = 1.f;
                    }
            }
        conditioned.release();
        capture("patches", patches);
        progress("projecting", 3);
        auto video = execute("patch-in", {patches}).at(0);
        expect(video, {gh*gw, 2560}, "patch-in");
        capture("patch-in", video);
        patches.release();
        auto text = package.text;
        capture("text-in", text); capture("time-in", package.time);
        video = video.reshape(2560, gw, gh, 1);
        for (int index = 0; index < 32; ++index) {
            const auto name = "block-"+std::string(index < 10 ? "0" : "")+std::to_string(index);
            progress(name, 4+index);
            auto values = execute(name, {video, text, package.time});
            video = values[0]; text = values[1];
            expect(video, {1, gh, gw, 2560}, name);
            expect(text, {58, 2560}, name+" text");
            capture(name+"-video", video.reshape(2560, gh*gw));
            capture(name+"-text", text);
        }
        progress("denoising", 36);
        auto prediction = execute("patch-out", {video.reshape(2560, gh*gw)}).at(0);
        expect(prediction, {gh*gw, 64}, "patch-out");
        capture("patch-out", prediction);
        ncnn::Mat velocity(lw, lh, 16, size_t(4), 1), latent(lw, lh, 16, size_t(4), 1);
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx)
                        for (int c = 0; c < 16; ++c) {
                            const auto value = prediction.row(y*gw+x)[(dy*2+dx)*16+c];
                            velocity.channel(c).row(y*2+dy)[x*2+dx] = value;
                            latent.channel(c).row(y*2+dy)[x*2+dx] =
                                (noise.channel(c).row(y*2+dy)[x*2+dx]-value)/.9152f;
                        }
        capture("velocity", velocity); capture("latent", latent);
        video.release(); text.release(); prediction.release(); noise.release(); velocity.release();
        progress("decoding", 37);
        auto decoded = execute("decoder", {latent}).at(0);
        expect(decoded, {3, height, width}, "decoder");
        capture("decoded", decoded);
        check();
        save_image(request.output_directory/"output.partial.png", decoded);
        if (hash(request.input_file) != input_hash || hash(request.model_directory/"manifest.json") != package.identity)
            throw std::runtime_error("Input or model manifest changed during execution");
        check();
        Json report{{"schema_version", "seedvr2-image-run-v1"}, {"status", "SUCCEEDED"},
            {"build", SEEDVR2_BUILD_VERSION},
            {"profile", profile}, {"backend", request.vulkan ? "ncnn-vulkan" : "ncnn-cpu"},
            {"device", gpu_info}, {"ncnn_commit", SEEDVR2_NCNN_COMMIT},
            {"model_manifest_sha256", package.identity}, {"model_verified", false},
            {"numerical_validation", "NOT_PERFORMED_BY_RUNNER"}, {"model_certificate", nullptr},
            {"input", {{"sha256", input_hash}, {"width", image.width}, {"height", image.height}}},
            {"output", {{"path", "output.png"}, {"sha256", hash(request.output_directory/"output.partial.png")},
                         {"width", width}, {"height", height}}},
            {"comparison_input", "comparison-input.png"}, {"seed", request.seed},
            {"noise_algorithm", "splitmix64-box-muller-v1"}, {"sampling", package.manifest.at("sampling")},
            {"preprocessing", "RGB8 / bicubic antialias / center crop 16 / normalize [-1,1]"},
            {"host_operations", {"image IO", "resizing", "layouts", "posterior sampling", "Euler endpoint"}},
            {"dispatch", "EXPLICIT_PER_LAYER_NO_BACKEND_FALLBACK"}, {"stages", stages},
            {"diagnostics", diagnostics},
            {"total_ms", std::chrono::duration<double, std::milli>(Clock::now()-started).count()}};
        report["implementation"]=implementation_identity();
        if (report["implementation"].contains("executable_sha256")) report["executable_sha256"]=report["implementation"]["executable_sha256"];
        report["resources"]=process_resources();
        report["weight_io"]=request.mapped_weights?"mapped":"buffered";
        const auto &preflight=std::get<Preflight>(ready);
        report["resources"]["package_bytes"]=preflight.package_bytes;
        report["resources"]["largest_graph_weight_bytes"]=preflight.largest_graph_bytes;
        double loads=0,compute=0;
        for (const auto &stage:stages) {loads+=stage.at("load_ms").get<double>();compute+=stage.at("compute_ms").get<double>();}
        report["timing"]={{"package_validation_ms",validation_ms},{"graph_load_ms",loads},{"graph_compute_ms",compute},
            {"other_host_wait_and_cleanup_ms",report.at("total_ms").get<double>()-validation_ms-loads-compute}};
        report["package_authenticated"]=true;
        report["converter_ncnn_commit"]=package.manifest.at("ncnn_commit");

        const auto result = report.dump(2);
        std::ofstream file(request.output_directory/"run.partial.json");
        file << result << '\n'; file.close();
        if (!file) throw std::runtime_error("Cannot write image run report");
        check();
        std::filesystem::rename(request.output_directory/"output.partial.png", request.output_directory/"output.png");
        std::filesystem::rename(request.output_directory/"run.partial.json", request.output_directory/"run.json");
        if (observer) observer({"completed", 38, 38,
            std::chrono::duration<double, std::milli>(Clock::now()-started).count()});
        return result;
    } catch (const Cancelled &e) { return Error{"CANCELLED", "run", e.what()}; }
      catch (const std::exception &e) { return Error{"IMAGE_EXECUTION_FAILED", "run", e.what()}; }
}
} // namespace seedvr2::engine
