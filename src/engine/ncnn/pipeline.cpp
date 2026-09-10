#include "seedvr2/pipeline.hpp"
#include "seedvr2/engine.hpp"
#include "seedvr2/video.hpp"
#include "package.hpp"
#include "video_io.hpp"
#include "memory.hpp"
#include "fp32_device.hpp"
#include <bit>
#if defined(__unix__)
#include <unistd.h>
#endif

namespace seedvr2 {
using namespace engine::detail;
using namespace engine::inference;
namespace {
bool is_video(MediaKind kind) {
    if (kind!=MediaKind::image && kind!=MediaKind::video) throw std::runtime_error("Unknown media kind");
    return kind==MediaKind::video;
}
std::filesystem::path existing_parent(std::filesystem::path path) {
    path=std::filesystem::absolute(path);
    while (!std::filesystem::exists(path)) {
        if (path==path.root_path()) throw std::runtime_error("No accessible output parent");
        path=path.parent_path();
    }
    if (!std::filesystem::is_directory(path)) throw std::runtime_error("Output parent is not a directory");
    return path;
}
PackageInfo package_info(const PackageManifest &p) {
    return {p.identity,Json{{"schema_version","seedvr2-package-verification-v1"},{"status","AUTHENTICATED_AND_INTEGRITY_CHECKED"},
        {"manifest_sha256",p.identity},{"profile",p.document.at("profile")},{"bytes",p.bytes},
        {"graphs",36},{"converter_ncnn_commit",p.document.at("ncnn_commit")},
        {"model_verified",false},{"quality_certification",false}}.dump(2),p.bytes};
}
}

Result<Preflight> preflight(const RestoreRequest &r) {
    std::string field="parameters";
    try {
        const bool video=is_video(r.kind);
        memory::validate(r.memory,r.backend==Backend::vulkan);
        if (r.weight_io!=WeightIO::buffered && r.weight_io!=WeightIO::mapped) throw std::runtime_error("Unknown weight reader");
#if !defined(__linux__)
        if (r.weight_io==WeightIO::mapped) throw std::runtime_error("Mapped weight loading requires Linux");
#endif
        if constexpr (std::endian::native!=std::endian::little) throw std::runtime_error("The FP32 package requires a little endian host");
        if ((r.backend!=Backend::cpu && r.backend!=Backend::vulkan) || r.threads<1 || r.threads>32 || r.gpu_index< -1 ||
            r.long_side<64 || r.long_side>(video?128:512) || r.long_side%16 || (video && (r.max_frames<1 || r.max_frames>17)))
            throw std::runtime_error("Unsupported parameters: image size 64..512, video size 64..128 and 1..17 frames; size must be divisible by 16, threads 1..32");
        field="output";
        if (r.output_directory.empty()) throw std::runtime_error("Choose a new output directory");
        if (std::filesystem::exists(r.output_directory) &&
            (!std::filesystem::is_directory(r.output_directory) || !std::filesystem::is_empty(r.output_directory)))
            throw std::runtime_error("Output directory must be new or empty");
        const auto parent=existing_parent(r.output_directory);
#if defined(__unix__)
        if (access(parent.c_str(),W_OK|X_OK)!=0) throw std::runtime_error("Output parent is not writable");
#endif
        field="input";
        const auto pixels=video?load_video(r.input_file,1).frames.front():load_image(r.input_file);
        const auto prepared=prepare_image(pixels,r.long_side);
        const int frames=video?r.max_frames:1, lt=video?(frames-1+3)/4+1:1;
        const auto tokens=std::uint64_t(lt)*std::uint64_t(prepared.w/16)*std::uint64_t(prepared.h/16);
        // Upper bound for persisted intermediates and uncompressed output copies,
        // not a GPU memory estimate or a claim about codec compression.
        const auto reserve=8ULL*1024*1024+std::uint64_t(prepared.w)*std::uint64_t(prepared.h)*std::uint64_t(frames)*24+
            (r.diagnostic_tensors?66*(tokens+58)*2560*4+16ULL*1024*1024:0);
        field="output";
        if (std::filesystem::space(parent).available<reserve) throw std::runtime_error("Insufficient output disk space for the requested artifacts");
        field="device";
        Json device=nullptr;
        if (r.backend==Backend::vulkan) {
            std::lock_guard lock(engine_mutex); init_gpu();
            const int index=r.gpu_index<0?ncnn::get_default_gpu_index():r.gpu_index;
            if (index<0 || index>=ncnn::get_gpu_count()) throw std::runtime_error("Requested Vulkan device is unavailable; select an available device or explicitly choose CPU");
            device=device_info(index);
            device["fp32_b_arithmetic"]=require_fp32_device(index);
        }
        field="model";
        const auto p=inspect_manifest(r.model_directory,video);
        Json report{{"schema_version","seedvr2-preflight-v1"},{"status","READY_FOR_WEIGHT_VERIFICATION"},
            {"kind",video?"video":"image"},{"backend",r.backend==Backend::vulkan?"vulkan":"cpu"},{"device",device},
            {"output",{{"width",prepared.w},{"height",prepared.h},{"maximum_frames",frames},{"reserve_bytes",reserve}}},
            {"model",{{"manifest_sha256",p.identity},{"profile",p.document.at("profile")},{"package_bytes",p.bytes},
                {"largest_graph_bytes",p.largest_graph_bytes},{"manifest_authenticated",true},{"weight_hashes_verified",false}}},
            {"model_verified",false}};
        report["memory"]={{"weight_policy",memory::name(r.memory.weights)},
            {"gpu_reserve_bytes",r.memory.gpu_reserve_bytes},{"scope","rechecked before each graph; not a peak-memory guarantee"}};
        return Preflight{prepared.w,prepared.h,frames,p.bytes,p.largest_graph_bytes,reserve,p.identity,report.dump(2)};
    } catch (const std::exception &e) { return Error{"PREFLIGHT_FAILED",field,e.what()}; }
}

Result<RunResult> restore(const RestoreRequest &r, ProgressObserver observer, CancellationCheck cancelled) {
    if (cancelled && cancelled()) return Error{"CANCELLED","run","Processing cancelled"};
    try {
        const bool video=is_video(r.kind);
        engine::VideoRequest native;
        native.model_directory=r.model_directory; native.input_file=r.input_file; native.output_directory=r.output_directory;
        native.vulkan=r.backend==Backend::vulkan; native.gpu_index=r.gpu_index; native.threads=r.threads;
        native.long_side=r.long_side; native.max_frames=r.max_frames; native.seed=r.seed; native.diagnostic_tensors=r.diagnostic_tensors;
        native.mapped_weights=r.weight_io==WeightIO::mapped;
        native.memory=r.memory;
        if (r.weight_io!=WeightIO::buffered && r.weight_io!=WeightIO::mapped) return Error{"PREFLIGHT_FAILED","weight_io","Unknown weight reader"};
        if (r.backend!=Backend::cpu && r.backend!=Backend::vulkan) return Error{"PREFLIGHT_FAILED","backend","Unknown backend"};
        auto value=video?engine::run_video(native,observer,cancelled):engine::run_image(native,observer,cancelled);
        if (is_error(value)) return std::get<Error>(value);
        auto text=std::get<std::string>(std::move(value)); const auto report=Json::parse(text);
        return RunResult{r.output_directory/(video?"output.mp4":"output.png"),r.output_directory/"run.json",
            report.at("output").at("width"),report.at("output").at("height"),video?report.at("output").at("frames").get<int>():1,
            report.at("total_ms"),std::move(text)};
    } catch (const std::exception &e) { return Error{"EXECUTION_FAILED","run",e.what()}; }
}

Result<PackageInfo> verify_model(const std::filesystem::path &directory, MediaKind kind,
                                 ProgressObserver observer, CancellationCheck cancelled) {
    try {
        const auto check=[&] {if (cancelled && cancelled()) throw Cancelled();}; check();
        Package verified(directory,check,is_video(kind),observer);
        auto p=inspect_manifest(directory,is_video(kind));
        if (p.identity!=verified.identity) throw std::runtime_error("Model changed during verification");
        return package_info(p);
    } catch (const Cancelled &e) { return Error{"CANCELLED","model",e.what()}; }
      catch (const std::exception &e) { return Error{"MODEL_PACKAGE_INVALID","model",e.what()}; }
}

Result<PackageInfo> copy_model(const std::filesystem::path &source, const std::filesystem::path &destination,
                               MediaKind kind, ProgressObserver observer, CancellationCheck cancelled) {
    std::filesystem::path temporary;
    bool owned=false;
    try {
        if (destination.empty() || std::filesystem::exists(destination)) throw std::runtime_error("Model destination must be new");
        const auto checked=verify_model(source,kind,observer,cancelled);
        if (is_error(checked)) return std::get<Error>(checked);
        const auto metadata=inspect_manifest(source,is_video(kind));
        if (std::filesystem::space(existing_parent(destination.parent_path().empty()?".":destination.parent_path())).available<metadata.bytes+4*1024*1024)
            throw std::runtime_error("Insufficient disk space for the model copy");
        std::filesystem::create_directories(destination.parent_path().empty()?".":destination.parent_path());
        temporary=destination; temporary+=".partial-"+std::to_string(Clock::now().time_since_epoch().count());
        if (!std::filesystem::create_directory(temporary)) throw std::runtime_error("Cannot create model copy staging directory");
        owned=true;
        std::vector<std::filesystem::path> files{"manifest.json"};
        for (const auto &row : metadata.document.at("graphs"))
            for (const auto *key : {"param","weights"}) files.push_back(utf8_path(row.at(key).at("path").get<std::string>()));
        for (const auto *key : {"text","time"}) files.push_back(utf8_path(metadata.document.at("constants").at(key).at("path").get<std::string>()));
        const auto start=Clock::now(); int done=0;
        for (const auto &file : files) {
            if (cancelled && cancelled()) throw Cancelled();
            std::filesystem::create_directories((temporary/file).parent_path());
            if (!std::filesystem::copy_file(source/file,temporary/file)) throw std::runtime_error("Model file copy failed");
            if (observer) observer({"copying-model",++done,static_cast<int>(files.size()),std::chrono::duration<double,std::milli>(Clock::now()-start).count()});
        }
        auto result=verify_model(temporary,kind,observer,cancelled);
        if (is_error(result)) {std::filesystem::remove_all(temporary); return std::get<Error>(result);}
        if (std::get<PackageInfo>(result).manifest_sha256!=metadata.identity) throw std::runtime_error("Model identity changed during copying");
        std::filesystem::rename(temporary,destination); owned=false; return result;
    } catch (const std::exception &e) {
        if (owned) {std::error_code ignored; std::filesystem::remove_all(temporary,ignored);}
        return Error{dynamic_cast<const Cancelled *>(&e)?"CANCELLED":"MODEL_COPY_FAILED","model",e.what()};
    }
}

std::string build_info() { return engine::build_status(); }
} // namespace seedvr2
