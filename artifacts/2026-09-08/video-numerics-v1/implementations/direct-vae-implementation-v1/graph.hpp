#pragma once
#include "seedvr2/path.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <command.h>
#include <fstream>
#include <functional>
#include <gpu.h>
#include <iomanip>
#include <memory>
#include <mutex>
#include <net.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
namespace seedvr2::engine::detail {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
inline std::mutex engine_mutex;
inline void init_gpu() {
    static std::once_flag once;
    static int result = -1;
    std::call_once(once, [] { result = ncnn::create_gpu_instance(); });
    if (result != 0)
        throw std::runtime_error("Cannot initialize Vulkan");
}
inline std::string hash(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot read artifact");
    auto ctx =
        std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("Hash initialization failed");
    std::array<char, 65536> buffer{};
    while (in.read(buffer.data(), buffer.size()) || in.gcount())
        if (EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<std::size_t>(in.gcount())) != 1)
            throw std::runtime_error("Hash update failed");
    if (in.bad())
        throw std::runtime_error("Artifact read failed");
    std::array<unsigned char, 32> digest{};
    unsigned size = 0;
    if (EVP_DigestFinal_ex(ctx.get(), digest.data(), &size) != 1 || size != 32)
        throw std::runtime_error("Hash failed");
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto x : digest)
        out << std::setw(2) << static_cast<unsigned>(x);
    return out.str();
}
inline std::string hash_text(std::string_view text) {
    std::array<unsigned char,32> digest{};
    unsigned size=0;
    if (EVP_Digest(text.data(),text.size(),digest.data(),&size,EVP_sha256(),nullptr)!=1 || size!=32)
        throw std::runtime_error("Content hash failed");
    std::ostringstream out;out << std::hex << std::setfill('0');
    for (auto x:digest) out << std::setw(2) << static_cast<unsigned>(x);
    return out.str();
}
inline std::filesystem::path artifact(const std::filesystem::path &root, const Json &row,
                               std::uint64_t max_bytes = 256ULL * 1024 * 1024, bool verify_hash = true) {
    const auto name = row.at("path").get<std::string>();
    const auto p = seedvr2::utf8_path(name);
    if (p.empty() || p.is_absolute() || name.find('\\') != std::string::npos)
        throw std::runtime_error("Artifact must have a relative path");
    auto resolved = root;
    for (const auto &part : p) {
        if (part == ".." || part == ".")
            throw std::runtime_error("Invalid artifact path");
        resolved /= part;
        if (std::filesystem::is_symlink(resolved))
            throw std::runtime_error("Artifact symlinks are not supported");
    }
    if (!std::filesystem::is_regular_file(resolved) ||
        std::filesystem::file_size(resolved) > max_bytes ||
        (row.contains("bytes") && row.at("bytes") != std::filesystem::file_size(resolved)))
        throw std::runtime_error("Missing or oversized artifact");
    if (verify_hash && hash(resolved) != row.at("sha256").get<std::string>())
        throw std::runtime_error("Artifact hash mismatch");
    return resolved;
}
inline Json read_document(const std::filesystem::path &path) {
    if (std::filesystem::file_size(path) > 65536)
        throw std::runtime_error("Case manifest exceeds 64 KiB");
    std::ifstream input(path);
    std::vector<std::set<std::string>> keys;
    auto callback = [&](int depth, Json::parse_event_t event, Json &value) {
        if (depth > 16)
            throw std::runtime_error("Case nesting exceeds 16 levels");
        if (event == Json::parse_event_t::object_start)
            keys.emplace_back();
        else if (event == Json::parse_event_t::object_end)
            keys.pop_back();
        else if (event == Json::parse_event_t::key &&
                 !keys.back().insert(value.get<std::string>()).second)
            throw std::runtime_error("Duplicate case field");
        return true;
    };
    return Json::parse(input, callback);
}
inline int integer(const Json &value, int lower, int upper) {
    if (!value.is_number_integer() || value < lower || value > upper)
        throw std::runtime_error("Case requires an integer within supported bounds");
    return value.get<int>();
}
inline std::array<int, 3> dimensions(const Json &value, std::array<int, 3> upper) {
    if (!value.is_array() || value.size() != 3)
        throw std::runtime_error("Expected three tensor dimensions");
    return {integer(value[0], 1, upper[0]), integer(value[1], 1, upper[1]),
            integer(value[2], 1, upper[2])};
}
inline Json read_case(const std::filesystem::path &path) {
    auto doc = read_document(path);
    if (doc.at("schema_version") != "awa-case-v1" || doc.at("precision") != "fp32" ||
        doc.at("reference_profile") != "FP32-B")
        throw std::runtime_error("Unsupported AWA case schema or precision");
    return doc;
}
inline void read_tensor(const std::filesystem::path &path, ncnn::Mat &mat) {
    const std::size_t channel_count = mat.dims >= 3 ? static_cast<std::size_t>(mat.c) : 1;
    const std::size_t elements = static_cast<std::size_t>(mat.w) * mat.h * mat.d;
    if (std::filesystem::file_size(path) != channel_count * elements * sizeof(float))
        throw std::runtime_error("Input tensor length mismatch");
    std::ifstream in(path, std::ios::binary);
    for (std::size_t c = 0; c < channel_count; ++c) {
        auto *p = static_cast<float *>(mat.data) + c * mat.cstep;
        in.read(reinterpret_cast<char *>(p),
                static_cast<std::streamsize>(elements * sizeof(float)));
        if (!in)
            throw std::runtime_error("Input tensor read failed");
        for (std::size_t i = 0; i < elements; ++i)
            if (!std::isfinite(p[i]))
                throw std::runtime_error("Non-finite input tensor");
    }
}
inline Json write_tensor(const std::filesystem::path &directory, const char *name, const ncnn::Mat &mat) {
    if (mat.empty() || mat.n != 1 || mat.elempack != 1 || mat.elemsize != 4)
        throw std::runtime_error("Unexpected output tensor representation");
    const auto path = directory / name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::size_t channels = mat.dims >= 3 ? static_cast<std::size_t>(mat.c) : 1;
    const std::size_t elements = static_cast<std::size_t>(mat.w) * mat.h * mat.d;
    for (std::size_t c = 0; c < channels; ++c) {
        const auto *p = static_cast<const float *>(mat.data) + c * mat.cstep;
        for (std::size_t i = 0; i < elements; ++i)
            if (!std::isfinite(p[i]))
                throw std::runtime_error("Non-finite graph output");
        out.write(reinterpret_cast<const char *>(p),
                  static_cast<std::streamsize>(elements * sizeof(float)));
    }
    out.close();
    if (!out)
        throw std::runtime_error("Output write failed");
    return Json{{"path", name},
                {"sha256", hash(path)},
                {"dtype", "f32le"},
                {"elements", channels * elements}};
}
inline Json device_info(int index) {
    const auto &info = ncnn::get_gpu_info(index);
    return Json{{"index", index},
                {"name", info.device_name()},
                {"vendor_id", info.vendor_id()},
                {"device_id", info.device_id()},
                {"driver_version", info.driver_version()},
                {"api_version", info.api_version()},
                {"subgroup_size", info.subgroup_size()}};
}

// Diagnostic execution deliberately dispatches each layer's requested overload.
// An unsupported Vulkan layer fails here; no Extractor CPU fallback is available.
template <typename Mat>
std::vector<Mat> graph_forward(ncnn::Net &net, const std::vector<Mat> &inputs, ncnn::VkCompute *command,
                  const ncnn::Option &opt, Json &trace,
                  const std::function<void()> &checkpoint = {}, int submit_interval = 0,
                  const std::function<void(const ncnn::Layer &, const std::vector<Mat> &)> &observe = {}) {
    std::vector<Mat> blobs(net.blobs().size());
    std::vector<unsigned> uses(blobs.size());
    for (const auto *layer : net.layers())
        for (const auto index : layer->bottoms)
            ++uses.at(index);
    for (const auto index : net.output_indexes())
        ++uses.at(index);
    if (inputs.size() != net.input_indexes().size())
        throw std::runtime_error("Graph input arity mismatch");
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto name = "in" + std::to_string(i);
        bool bound = false;
        for (const auto index : net.input_indexes())
            if (net.blobs().at(index).name == name) {
                blobs.at(index) = inputs[i];
                bound = true;
            }
        if (!bound)
            throw std::runtime_error("Missing named graph input: " + name);
    }
    int pending_layers = 0;
    for (const auto *layer : net.layers()) {
        if (layer->type == "Input")
            continue;
        if (checkpoint)
            checkpoint();
        std::vector<Mat> bottoms;
        for (const auto index : layer->bottoms) {
            Mat value = blobs.at(index);
            if (value.empty())
                throw std::runtime_error("Graph is not topologically ordered");
            if constexpr (std::is_same_v<Mat, ncnn::VkMat>) {
                if (!layer->support_vulkan)
                    throw std::runtime_error("Vulkan implementation unavailable: " + layer->type);
                int pack = 1;
                const int count = value.elempack * (value.dims == 1 ? value.w :
                                                   value.dims == 2 ? value.h : value.c);
                if (layer->support_vulkan_packing) {
                    pack = count % 4 == 0 ? 4 : 1;
                    if (layer->support_vulkan_any_packing)
                        pack = value.elempack;
                }
                if (value.elempack != pack) {
                    Mat converted;
                    net.vulkan_device()->convert_packing(value, converted, pack, *command, opt);
                    value = converted;
                }
            }
            if (value.empty())
                throw std::runtime_error("Graph layout conversion failed");
            bottoms.push_back(value);
        }
        std::vector<Mat> tops(layer->tops.size());
        int result = -1;
        if (layer->one_blob_only && (bottoms.size() != 1 || tops.size() != 1))
            throw std::runtime_error("Invalid single-input layer arity");
        if constexpr (std::is_same_v<Mat, ncnn::VkMat>) {
            result = layer->one_blob_only
                         ? layer->forward(bottoms[0], tops[0], *command, opt)
                         : layer->forward(bottoms, tops, *command, opt);
        } else {
            result = layer->one_blob_only ? layer->forward(bottoms[0], tops[0], opt)
                                          : layer->forward(bottoms, tops, opt);
        }
        if (result != 0)
            throw std::runtime_error("Layer execution failed: " + layer->name + " (" +
                                     layer->type + ")");
        for (std::size_t i = 0; i < tops.size(); ++i) {
            if (tops[i].empty())
                throw std::runtime_error("Empty intermediate: " + layer->name);
            blobs.at(layer->tops[i]) = tops[i];
        }
        trace.push_back({{"name", layer->name}, {"type", layer->type},
                         {"backend", std::is_same_v<Mat, ncnn::VkMat> ? "vulkan" : "cpu"}});
        if (observe)
            observe(*layer, tops);
        for (const auto index : layer->bottoms)
            if (--uses.at(index) == 0)
                blobs.at(index).release();
        if constexpr (std::is_same_v<Mat, ncnn::VkMat>) {
            // A bounded command batch releases completed GPU references and
            // provides cancellation points for the high-resolution VAE.
            if (submit_interval > 0 && ++pending_layers >= submit_interval) {
                if (command->submit_and_wait() != 0 || command->reset() != 0)
                    throw std::runtime_error("Vulkan command batch failed");
                pending_layers = 0;
            }
        }
    }
    std::vector<Mat> outputs;
    for (std::size_t i = 0; i < net.output_indexes().size(); ++i) {
        const auto name = "out" + std::to_string(i);
        bool bound = false;
        for (const auto index : net.output_indexes())
            if (net.blobs().at(index).name == name) {
                outputs.push_back(blobs.at(index));
                bound = true;
            }
        if (!bound)
            throw std::runtime_error("Missing named graph output: " + name);
    }
    return outputs;
}
struct VulkanAllocators {
    explicit VulkanAllocators(const ncnn::VulkanDevice *device) : device(device) {
        blob = device->acquire_blob_allocator();
        staging = device->acquire_staging_allocator();
    }
    ~VulkanAllocators() {
        if (blob)
            device->reclaim_blob_allocator(blob);
        if (staging)
            device->reclaim_staging_allocator(staging);
    }
    const ncnn::VulkanDevice *device;
    ncnn::VkAllocator *blob = nullptr, *staging = nullptr;
};
inline ncnn::Mat allocate_tensor(const std::vector<int> &shape) {
    if (shape.size() == 2)
        return ncnn::Mat(shape[1], shape[0], size_t(4), 1);
    if (shape.size() == 3)
        return ncnn::Mat(shape[2], shape[1], shape[0], size_t(4), 1);
    if (shape.size() == 4)
        return ncnn::Mat(shape[3], shape[2], shape[1], shape[0], size_t(4), 1);
    throw std::runtime_error("Unsupported diagnostic tensor rank");
}
inline std::vector<int> tensor_shape(const ncnn::Mat &mat) {
    if (mat.dims == 2)
        return {mat.h, mat.w};
    if (mat.dims == 3)
        return {mat.c, mat.h, mat.w};
    if (mat.dims == 4)
        return {mat.c, mat.d, mat.h, mat.w};
    return {};
}
}
