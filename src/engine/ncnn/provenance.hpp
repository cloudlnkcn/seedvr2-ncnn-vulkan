#pragma once
#include "graph.hpp"
#include "seedvr2/pipeline.hpp"
#if defined(__linux__)
#include <dlfcn.h>
#include <sys/resource.h>
#endif
namespace seedvr2::engine::detail {
inline Json implementation_identity() {
    Json result=Json::object();
#if defined(__linux__)
    result["executable_sha256"]=hash("/proc/self/exe");
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(&seedvr2::build_info),&info) || !info.dli_fname)
        throw std::runtime_error("Cannot locate the loaded SDK library for provenance");
    result["sdk_library_sha256"]=hash(info.dli_fname);
#endif
    return result;
}
inline Json process_resources() {
    Json report{{"weight_residency","one graph at a time"},
        {"autoregressive_kv_cache",false},{"temporal_vae_cache",false},
        {"pipeline_cache","per-run Vulkan shader pipelines; no model weights or autoregressive keys"},
        {"activation_peak_bytes",nullptr},{"workspace_peak_bytes",nullptr},{"file_cache_peak_bytes",nullptr},
        {"unmeasured_scope","Allocator categories are not inferred from process RSS or whole-card VRAM"}};
#if defined(__linux__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF,&usage)==0) {
        report["process_lifetime_peak_rss_bytes"]=std::uint64_t(usage.ru_maxrss)*1024;
        report["process_user_cpu_seconds"]=double(usage.ru_utime.tv_sec)+double(usage.ru_utime.tv_usec)/1000000.;
        report["process_system_cpu_seconds"]=double(usage.ru_stime.tv_sec)+double(usage.ru_stime.tv_usec)/1000000.;
    }
#endif
    return report;
}
} // namespace seedvr2::engine::detail
