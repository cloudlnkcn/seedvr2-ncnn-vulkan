#include "seedvr2/application.hpp"
#include "seedvr2/engine.hpp"
#include "seedvr2/image.hpp"
#include "seedvr2/video.hpp"
#include "seedvr2/path.hpp"
#include "seedvr2/protocol.hpp"
#include "seedvr2/validation.hpp"
#include <CLI/CLI.hpp>
#include <charconv>
#include <csignal>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {
using namespace seedvr2;
volatile std::sig_atomic_t cancel_requested = 0;
void request_cancel(int) { cancel_requested = 1; }
void progress(const Progress &p) {
    std::cerr << nlohmann::json({{"event","progress"},{"stage",p.stage},{"completed",p.completed},
        {"total",p.total},{"elapsed_ms",p.elapsed_ms}}).dump() << std::endl;
}
bool cancelled() { return cancel_requested!=0; }
int fail(const Error &error, int code = 2) {
    std::cout << serialize_error(error) << '\n';
    return code;
}
Result<std::string> read_request(const std::string &path) {
    std::ifstream file(seedvr2::utf8_path(path), std::ios::binary);
    if (!file)
        return Error{"FILE_READ", "request", "Cannot open planning request"};
    std::string text(max_request_bytes + 1, '\0');
    file.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(file.gcount()));
    if (file.bad())
        return Error{"FILE_READ", "request", "Failed reading planning request"};
    if (text.size() > max_request_bytes)
        return Error{"REQUEST_TOO_LARGE", "request", "Planning request exceeds 1 MiB"};
    return text;
}
int print(const Result<std::string> &result, int success = 0) {
    if (is_error(result))
        return fail(std::get<Error>(result));
    std::cout << std::get<std::string>(result) << '\n';
    return success;
}
int main_impl(int argc, char **argv) {
    CLI::App cli{"SeedVR2 native image restoration and independent diagnostics CLI."};
    cli.require_subcommand(1,1);
    std::signal(SIGINT,request_cancel);
    std::signal(SIGTERM,request_cancel);
    std::string database;
    cli.add_option("--database", database,
                   "SQLite workspace (optional; commands that do not save stay stateless)");
    auto db = [&] {
        return database.empty() ? default_database_path() : seedvr2::utf8_path(database);
    };
    auto version = cli.add_subcommand("version", "Print build and model status");
    auto caps = cli.add_subcommand("capabilities", "Machine-readable supported operations");
    auto run = cli.add_subcommand("run", "Restore one image using the complete native ncnn model");
    std::string image_model, image_input, image_output, image_backend = "vulkan";
    int image_size = 256, image_gpu = -1, image_threads = 4;
    std::uint64_t image_seed = 666;
    bool image_diagnostics = false, check_only = false;
    std::string image_weight_io="buffered",diagnostic_weight_io="buffered";
    std::string weight_placement="auto";
    std::uint64_t gpu_reserve_mib=0;
    run->add_option("--model", image_model, "Directory containing the complete model package")->required();
    run->add_option("--input", image_input, "PNG or JPEG input")->required();
    run->add_option("--output", image_output, "New output directory")->required();
    run->add_option("--size", image_size, "Output long side, 64..512, divisible by 16")->check(CLI::Range(64, 512));
    run->add_option("--backend", image_backend)->check(CLI::IsMember({"cpu", "vulkan"}));
    run->add_option("--gpu", image_gpu)->check(CLI::Range(-1, 64));
    run->add_option("--threads", image_threads)->check(CLI::Range(1, 32));
    run->add_option("--seed", image_seed);
    run->add_flag("--diagnostic-tensors", image_diagnostics, "Save intermediate tensors for independent reference comparison");
    run->add_flag("--check",check_only,"Check parameters, input, model manifest, output space and device without inference");
    auto video_run=cli.add_subcommand("run-video", "Restore the first 1..17 frames jointly with the temporal VAE and 3D attention");
    int video_frames=17,video_size=128;
    video_run->add_option("--model",image_model)->required();
    video_run->add_option("--input",image_input,"Local MP4/WebM/MKV")->required();
    video_run->add_option("--output",image_output,"New output directory with playable MP4")->required();
    video_run->add_option("--frames",video_frames,"Maximum input frames; preview takes the start of the video")->check(CLI::Range(1,17));
    video_run->add_option("--size",video_size,"Output long side, 64..128, divisible by 16")->check(CLI::Range(64,128));
    video_run->add_option("--backend",image_backend)->check(CLI::IsMember({"cpu","vulkan"}));
    video_run->add_option("--gpu",image_gpu)->check(CLI::Range(-1,64));
    video_run->add_option("--threads",image_threads)->check(CLI::Range(1,32));
    video_run->add_option("--seed",image_seed);
    video_run->add_flag("--diagnostic-tensors",image_diagnostics);
    video_run->add_flag("--check",check_only,"Check the requested short-video operation without inference");
    for (auto *command : {run,video_run})
        command->add_option("--weight-io",image_weight_io,"buffered (default) or experimental Linux mapped weights")->check(CLI::IsMember({"buffered","mapped"}));
    auto engine_cli = cli.add_subcommand("engine", "ncnn backend, AWA and checkpoint submodel diagnostics");
    engine_cli->require_subcommand(1);
    auto engine_status =
        engine_cli->add_subcommand("status", "Compiled ncnn operator capabilities");
    auto devices = engine_cli->add_subcommand("devices", "Probe actual local Vulkan devices");
    auto awa = engine_cli->add_subcommand("awa", "Run a hashed AWA test case through ncnn");
    auto graph = engine_cli->add_subcommand("graph", "Run a hashed VAE submodel case with explicit backend dispatch");
    auto block = engine_cli->add_subcommand("block", "Run a complete exported DiT block with explicit backend dispatch");
    auto self_test = engine_cli->add_subcommand("self-test", "Run embedded AWA numerical diagnostics offline");
    for (auto *command : {run,video_run,graph,block}) {
        command->add_option("--weights",weight_placement,"Vulkan weights: auto (default), device or host (RAM)")
            ->check(CLI::IsMember({"auto","device","host"}));
        command->add_option("--gpu-reserve-mib",gpu_reserve_mib,"Auto placement margin; 0 uses 1/4 of the current heap budget")
            ->check(CLI::Range(std::uint64_t(0),UINT64_MAX/(1024*1024)));
    }
    const auto memory_options=[&] {
        return MemoryOptions{weight_placement=="host"?WeightPlacement::host:
            weight_placement=="device"?WeightPlacement::device:WeightPlacement::automatic,
            gpu_reserve_mib*1024*1024};
    };
    for (auto *command : {graph,block})
        command->add_option("--weight-io",diagnostic_weight_io)->check(CLI::IsMember({"buffered","mapped"}));
    std::string case_file, output_directory, backend = "cpu";
    int gpu_index = -1, threads = 4;
    awa->add_option("--case", case_file)->required();
    awa->add_option("--output", output_directory)->required();
    awa->add_option("--backend", backend)->check(CLI::IsMember({"cpu", "vulkan"}));
    awa->add_option("--gpu", gpu_index);
    awa->add_option("--threads", threads)->check(CLI::Range(1, 32));
    graph->add_option("--case", case_file)->required();
    graph->add_option("--output", output_directory)->required();
    graph->add_option("--backend", backend)->check(CLI::IsMember({"cpu", "vulkan"}));
    graph->add_option("--gpu", gpu_index);
    graph->add_option("--threads", threads)->check(CLI::Range(1, 32));
    block->add_option("--case", case_file)->required();
    block->add_option("--output", output_directory)->required();
    block->add_option("--backend", backend)->check(CLI::IsMember({"cpu", "vulkan"}));
    block->add_option("--gpu", gpu_index);
    block->add_option("--threads", threads)->check(CLI::Range(1, 32));
    bool save_self_test = false;
    self_test->add_option("--backend", backend)->check(CLI::IsMember({"cpu", "vulkan"}));
    self_test->add_option("--gpu", gpu_index)->check(CLI::Range(-1, 64));
    self_test->add_flag("--save", save_self_test, "Save the actual diagnostic result for the local Web workspace");
    auto plan = cli.add_subcommand("plan", "Evaluate declared geometry with the shared C++ core");
    std::string request;
    bool save = false;
    plan->add_option("--request", request, "Planning request JSON")->required();
    plan->add_flag("--save", save, "Save a plan, without submitting inference");
    auto windows =
        cli.add_subcommand("windows", "Inspect official window geometry (no attention math)");
    std::vector<std::string> axes;
    bool shifted = false;
    windows->add_option("--tokens", axes, "T H W")->expected(3)->required();
    windows->add_flag("--shifted", shifted);
    auto models =
        cli.add_subcommand("models", "Model policy, missing evidence and raw artifact audits");
    models->require_subcommand(1);
    auto verify=models->add_subcommand("verify","Authenticate the reviewed model package and hash every file offline");
    auto copy=models->add_subcommand("copy","Copy and verify a model package for offline relocation");
    std::string package_kind="image",package_destination;
    for (auto *command : {verify,copy}) {
        command->add_option("--model",image_model,"Source model package")->required();
        command->add_option("--kind",package_kind)->check(CLI::IsMember({"image","video"}));
    }
    copy->add_option("--output",package_destination,"New destination model directory")->required();
    auto status = models->add_subcommand("status", "Current model validation status");
    auto policy =
        models->add_subcommand("policy", "Exact acceptance policy and calibration status");
    auto audit = models->add_subcommand(
        "audit",
        "Rehash local bundle files and compare raw f32 tensors; never imports a PASS claim");
    std::string bundle;
    bool audit_save = false;
    audit->add_option("--bundle", bundle, "Directory containing manifest.json and evidence files")
        ->required();
    audit->add_flag("--save", audit_save, "Persist computed report for the local Web workspace");
    auto history = cli.add_subcommand("history", "Read saved plans or model audits");
    history->require_subcommand(1);
    auto list = history->add_subcommand("list", "List latest 100 records");
    std::string kind = "plan", record_id;
    list->add_option("--kind", kind)->check(CLI::IsMember({"plan", "model-audit", "operator-test"}));
    auto get = history->add_subcommand("get", "Read one saved record");
    get->add_option("--id", record_id)->required();
    if (argc==1) {std::cout << cli.help() << '\n'; return 0;}
    try {
        cli.parse(argc, argv);
    } catch (const CLI::CallForHelp &e) {
        return cli.exit(e);
    } catch (const CLI::ParseError &) {
        return fail({"CLI_USAGE", "command", "Unknown or missing arguments. Use --help"});
    }
    if (*version) {
        std::cout << "0.7.0-native-preview; sdk=installed-cpp; ncnn=linked; "
                     "awa=cpu+vulkan; restoration=image+short-video; model=not-certified\n";
        return 0;
    }
    if (*caps) {
        std::cout << application_capabilities() << '\n';
        return 0;
    }
    if (*engine_status) {
        std::cout << engine::build_status() << '\n';
        return 0;
    }
    if (*devices)
        return print(engine::devices());
    if (*awa)
        return print(
            engine::run_awa({seedvr2::utf8_path(case_file), seedvr2::utf8_path(output_directory),
                             backend == "vulkan", gpu_index, threads}));
    if (*graph)
        return print(engine::run_graph({seedvr2::utf8_path(case_file),
                                       seedvr2::utf8_path(output_directory),
                                       backend == "vulkan", gpu_index, threads,diagnostic_weight_io=="mapped",memory_options()}));
    if (*block)
        return print(engine::run_dit_block({seedvr2::utf8_path(case_file),
                                           seedvr2::utf8_path(output_directory),
                                           backend == "vulkan", gpu_index, threads,diagnostic_weight_io=="mapped",memory_options()}));
    if (*self_test) {
        const auto result = save_self_test
            ? Application(db()).self_test_and_save("{\"backend\":\"" + backend + "\",\"gpu\":" + std::to_string(gpu_index) + "}")
            : engine::self_test(backend == "vulkan", gpu_index);
        if (is_error(result))
            return print(result);
        // PASS belongs to this numerical diagnostic only; never model certification.
        return print(result, nlohmann::json::parse(std::get<std::string>(result)).at("passed").get<bool>() ? 0 : 7);
    }
    if (*verify || *copy) {
        const auto media=package_kind=="video"?MediaKind::video:MediaKind::image;
        const auto result=*copy?copy_model(utf8_path(image_model),utf8_path(package_destination),media,progress,cancelled):
            verify_model(utf8_path(image_model),media,progress,cancelled);
        if (is_error(result)) {const auto &e=std::get<Error>(result);return fail(e,e.code=="CANCELLED"?130:2);}
        std::cout << std::get<PackageInfo>(result).report_json << '\n';return 0;
    }
    if (*run || *video_run) {
        const RestoreRequest settings{utf8_path(image_model),utf8_path(image_input),utf8_path(image_output),
            *video_run?MediaKind::video:MediaKind::image,image_backend=="vulkan"?Backend::vulkan:Backend::cpu,
            image_gpu,image_threads,*video_run?video_size:image_size,video_frames,image_seed,image_diagnostics,
            image_weight_io=="mapped"?WeightIO::mapped:WeightIO::buffered,memory_options()};
        if (check_only) {
            const auto result=preflight(settings);
            if (is_error(result)) return fail(std::get<Error>(result));
            std::cout << std::get<Preflight>(result).report_json << '\n';return 0;
        }
        const auto result=restore(settings,progress,cancelled);
        if (is_error(result)) {const auto &e=std::get<Error>(result);return fail(e,e.code=="CANCELLED"?130:2);}
        std::cout << std::get<RunResult>(result).report_json << '\n';return 0;
    }
    if (*plan) {
        const auto text = read_request(request);
        if (is_error(text))
            return fail(std::get<Error>(text), 3);
        if (save) {
            Application app(db());
            return print(app.save_plan(std::get<std::string>(text)));
        }
        return print(evaluate_planning_request(std::get<std::string>(text)));
    }
    if (*windows) {
        std::uint32_t dimensions[3]{};
        for (std::size_t i = 0; i < 3; ++i) {
            const auto parsed =
                std::from_chars(axes[i].data(), axes[i].data() + axes[i].size(), dimensions[i]);
            if (parsed.ec != std::errc{} || parsed.ptr != axes[i].data() + axes[i].size() ||
                dimensions[i] == 0)
                return fail(
                    {"INVALID_DIMENSION", "tokens", "Token axes must be positive uint32 integers"});
        }
        const auto result = plan_windows({dimensions[0], dimensions[1], dimensions[2]}, shifted);
        if (is_error(result))
            return fail(std::get<Error>(result));
        return print(serialize_windows(std::get<WindowPlan>(result)));
    }
    if (*status)
        return print(model_validation_status());
    if (*policy)
        return print(model_validation_policy());
    if (*audit) {
        // Exit 6 explicitly means no certificate; diagnostics may still be useful.
        if (audit_save) {
            Application app(db());
            return print(app.audit_and_save(seedvr2::utf8_path(bundle)), 6);
        }
        return print(audit_model_bundle(seedvr2::utf8_path(bundle)), 6);
    }
    if (*list || *get) {
        Application app(db());
        return print(*list ? app.records(kind) : app.record(record_id));
    }
    return fail({"CLI_USAGE", "command", "Use --help"});
}
} // namespace
int main(int argc, char **argv) {
    try {
        return main_impl(argc, argv);
    } catch (const std::exception &error) {
        return fail({"INTERNAL_ERROR", "", error.what()}, 5);
    }
}
