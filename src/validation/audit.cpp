#include "model_policy.hpp"
#include "seedvr2/path.hpp"
#include "seedvr2/validation.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <set>

namespace seedvr2 {
namespace {
using Json = nlohmann::json;
using Digest = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
std::string digest_end(EVP_MD_CTX *ctx) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(ctx, digest.data(), &length) != 1)
        throw std::runtime_error("SHA256 finalization failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned int i = 0; i < length; ++i) {
        result += hex[digest[i] >> 4];
        result += hex[digest[i] & 15];
    }
    return result;
}
Digest digest_begin() {
    Digest context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("SHA256 initialization failed");
    return context;
}
void exact(const Json &j, const std::set<std::string> &keys, const std::string &field) {
    if (!j.is_object())
        throw Error{"INVALID_TYPE", field, "Expected an object"};
    std::set<std::string> found;
    for (const auto &item : j.items())
        found.insert(item.key());
    if (keys != found)
        throw Error{"EVIDENCE_SCHEMA", field,
                    "Missing or unknown fields; verdicts are computed, not imported"};
}
std::string string(const Json &j, const std::string &field) {
    if (!j.is_string() || j.get_ref<const std::string &>().empty() ||
        j.get_ref<const std::string &>().size() > 1024)
        throw Error{"INVALID_TYPE", field, "Expected a nonempty string of at most 1024 bytes"};
    return j.get<std::string>();
}
bool is_sha(const std::string &value) {
    return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
}
Json strict_json(const std::string &text) {
    std::vector<std::set<std::string>> keys;
    return Json::parse(text, [&](int depth, Json::parse_event_t event, Json &value) {
        if (depth > 32)
            throw Error{"JSON_DEPTH", "manifest", "Maximum nesting is 32"};
        if (event == Json::parse_event_t::object_start)
            keys.emplace_back();
        if (event == Json::parse_event_t::object_end)
            keys.pop_back();
        if (event == Json::parse_event_t::key &&
            !keys.back().insert(value.get<std::string>()).second)
            throw Error{"DUPLICATE_KEY", "manifest", "Duplicate fields are rejected"};
        return true;
    });
}
std::filesystem::path contained_path(const std::filesystem::path &root, const std::string &name) {
    const std::filesystem::path relative = seedvr2::utf8_path(name);
    if (relative.empty() || relative.is_absolute() || relative.has_root_name() ||
        name.find('\\') != std::string::npos)
        throw Error{"ARTIFACT_PATH", name, "Use a relative bundle path"};
    auto current = root;
    for (const auto &part : relative) {
        if (part == ".." || part == ".")
            throw Error{"ARTIFACT_PATH", name, "Dot segments are not accepted"};
        current /= part;
        if (std::filesystem::is_symlink(std::filesystem::symlink_status(current)))
            throw Error{"ARTIFACT_PATH", name, "Symlinks are not accepted in evidence bundles"};
    }
    return current;
}
std::string read_manifest(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw Error{"FILE_READ", "manifest", "Cannot open manifest.json"};
    std::string text(1024 * 1024 + 1, '\0');
    file.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(file.gcount()));
    if (file.bad() || text.size() > 1024 * 1024)
        throw Error{"MANIFEST_SIZE", "manifest", "Cannot read manifest or it exceeds 1 MiB"};
    return text;
}
struct Artifact {
    std::filesystem::path path;
    std::string hash;
    bool valid;
};
float read_float(std::ifstream &stream) {
    std::array<unsigned char, 4> bytes{};
    stream.read(reinterpret_cast<char *>(bytes.data()), 4);
    if (stream.gcount() != 4)
        throw Error{"TENSOR_READ", "tensor", "Truncated tensor"};
    const std::uint32_t bits =
        static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
        (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
    return std::bit_cast<float>(bits);
}
Json compare(const Artifact &reference, const Artifact &candidate, std::uint64_t count) {
    if (std::filesystem::file_size(reference.path) != count * 4 ||
        std::filesystem::file_size(candidate.path) != count * 4)
        throw Error{"TENSOR_SIZE", "shape",
                    "Tensor files must contain exactly shape.product * 4 bytes"};
    std::ifstream a(reference.path, std::ios::binary), b(candidate.path, std::ios::binary);
    double sum_abs = 0, sum_square = 0, reference_square = 0, max_abs = 0;
    std::uint64_t nonfinite = 0, violations = 0, worst_index = 0;
    const auto tolerance = Json::parse(validation_data::policy).at("diagnostic_tolerance");
    const double atol = tolerance.at("atol"), rtol = tolerance.at("rtol");
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0)
        throw std::runtime_error("Invalid built-in diagnostic policy");
    for (std::uint64_t i = 0; i < count; ++i) {
        const double av = read_float(a), bv = read_float(b);
        if (!std::isfinite(av) || !std::isfinite(bv)) {
            ++nonfinite;
            continue;
        }
        const auto delta = std::abs(av - bv);
        if (delta > max_abs) {
            max_abs = delta;
            worst_index = i;
        }
        sum_abs += delta;
        sum_square += delta * delta;
        reference_square += av * av;
        if (delta > atol + rtol * std::abs(av))
            ++violations;
    }
    // Detect a tensor replaced or modified while the comparison was running.
    const auto after_a = sha256_file(reference.path), after_b = sha256_file(candidate.path);
    if (is_error(after_a) || is_error(after_b) ||
        std::get<std::string>(after_a) != reference.hash ||
        std::get<std::string>(after_b) != candidate.hash)
        throw Error{"ARTIFACT_CHANGED", "tensor", "Tensor changed during audit"};
    const double denominator = static_cast<double>(count);
    Json metrics = {{"elements", count},
                    {"nonfinite_pairs", nonfinite},
                    {"violations", violations},
                    {"atol", atol},
                    {"rtol", rtol},
                    {"worst_flat_index", worst_index}};
    // Partial metrics from only finite elements must never masquerade as complete metrics.
    metrics["max_abs"] = nonfinite ? Json(nullptr) : Json(max_abs);
    metrics["mae"] = nonfinite ? Json(nullptr) : Json(sum_abs / denominator);
    metrics["rmse"] = nonfinite ? Json(nullptr) : Json(std::sqrt(sum_square / denominator));
    metrics["nrmse"] =
        nonfinite || (reference_square == 0 && sum_square != 0)
            ? Json(nullptr)
            : Json(reference_square == 0 ? 0 : std::sqrt(sum_square / reference_square));
    metrics["diagnostic_status"] = nonfinite || violations ? "FAIL" : "PASS";
    metrics["certifies_model"] = false;
    return metrics;
}
Json empty_status() {
    const auto policy = Json::parse(validation_data::policy);
    Json gates = policy.at("gates");
    for (auto &gate : gates) {
        gate["status"] = "MISSING_EVIDENCE";
        gate["missing_artifact_roles"] = gate.at("required_artifact_roles");
    }
    return {{"schema_version", "1.0"},
            {"document_type", "model-validation-status"},
            {"model_id", "seedvr2-3b"},
            {"status", "BLOCKED"},
            {"model_verified", false},
            {"certificate", nullptr},
            {"policy_id", policy.at("policy_id")},
            {"policy_sha256", sha256_text(validation_data::policy)},
            {"calibration_status", "NOT_FROZEN"},
            {"gates", gates},
            {"blockers", Json::array({"ENGINE_NOT_BUILT", "REFERENCE_NOT_REGISTERED",
                                      "THRESHOLDS_NOT_FROZEN", "MODEL_GATES_NOT_EXECUTED"})}};
}
} // namespace

std::string sha256_text(std::string_view text) {
    auto ctx = digest_begin();
    if (EVP_DigestUpdate(ctx.get(), text.data(), text.size()) != 1)
        throw std::runtime_error("SHA256 update failed");
    return digest_end(ctx.get());
}
Result<std::string> sha256_file(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return Error{"FILE_READ", "artifact", "Cannot read evidence file"};
    auto ctx = digest_begin();
    std::array<char, 65536> buffer{};
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        if (EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<std::size_t>(file.gcount())) !=
            1)
            throw std::runtime_error("SHA256 update failed");
    }
    if (file.bad())
        return Error{"FILE_READ", "artifact", "Evidence read failed"};
    return digest_end(ctx.get());
}
std::string model_validation_policy() { return std::string(validation_data::policy); }
std::string model_validation_status() { return empty_status().dump(2); }

Result<std::string> audit_model_bundle(const std::filesystem::path &directory) {
    try {
        const auto root = std::filesystem::canonical(directory);
        const auto raw = read_manifest(contained_path(root, "manifest.json"));
        const auto manifest = strict_json(raw);
        exact(manifest, {"schema_version", "document_type", "scope", "artifacts", "tensor_pairs"},
              "manifest");
        if (manifest.at("schema_version") != "1.0" ||
            manifest.at("document_type") != "model-evidence-bundle")
            return Error{"SCHEMA_UNSUPPORTED", "manifest", "Expected model-evidence-bundle 1.0"};
        const std::set<std::string> scope_keys = {
            "model_id",      "weights_sha256", "export_sha256",  "runtime_build",
            "ncnn_commit",   "device",         "driver",         "precision",
            "shape_profile", "chunk_policy",   "dataset_sha256", "noise_sha256"};
        exact(manifest.at("scope"), scope_keys, "scope");
        if (manifest.at("scope").at("model_id") != "seedvr2-3b")
            return Error{"MODEL_UNSUPPORTED", "scope/model_id",
                         "This policy covers seedvr2-3b only"};
        Json missing_scope = Json::array();
        for (const auto &key : scope_keys) {
            const auto &value = manifest.at("scope").at(key);
            if (value.is_null()) {
                missing_scope.push_back(key);
                continue;
            }
            const auto v = string(value, key);
            if (key.ends_with("sha256") && !is_sha(v))
                return Error{"INVALID_HASH", key, "Expected lowercase SHA256"};
        }
        for (const auto key : {"artifacts", "tensor_pairs"})
            if (!manifest.at(key).is_array() || manifest.at(key).size() > 512)
                return Error{"EVIDENCE_SCHEMA", key, "Expected an array with at most 512 entries"};
        auto report = empty_status();
        report["document_type"] = "model-evidence-audit";
        report["scope"] = manifest.at("scope");
        report["scope_binding"] = "DECLARED_UNVERIFIED";
        report["missing_scope"] = missing_scope;
        report["manifest_sha256"] = sha256_text(raw);
        report["assurance"] = "Local file integrity and diagnostic comparison only; provenance and "
                              "suite coverage are not certified";
        report["artifact_checks"] = Json::array();
        report["tensor_checks"] = Json::array();
        std::set<std::string> roles, allowed_roles{"tensor"}, gate_ids;
        for (const auto &gate : report.at("gates")) {
            gate_ids.insert(gate.at("id").get<std::string>());
            for (const auto &role : gate.at("required_artifact_roles"))
                allowed_roles.insert(role.get<std::string>());
        }
        std::map<std::string, Artifact> artifacts;
        bool failed = false;
        for (const auto &item : manifest.at("artifacts")) {
            exact(item, {"id", "role", "path", "sha256"}, "artifacts");
            const auto id = string(item.at("id"), "id"), role = string(item.at("role"), "role");
            const auto hash = string(item.at("sha256"), "sha256");
            if (artifacts.contains(id) || !allowed_roles.contains(role) || !is_sha(hash))
                return Error{"EVIDENCE_SCHEMA", id,
                             "Duplicate artifact id, unknown role or invalid SHA256"};
            const auto path = contained_path(root, string(item.at("path"), "path"));
            const bool exists = std::filesystem::is_regular_file(path);
            const auto actual = exists
                                    ? sha256_file(path)
                                    : Result<std::string>(Error{"FILE_READ", id, "Missing file"});
            const bool valid = !is_error(actual) && std::get<std::string>(actual) == hash;
            artifacts.emplace(id, Artifact{path, hash, valid});
            Json row = {{"id", id},
                        {"role", role},
                        {"status", valid ? "HASH_MATCH" : (exists ? "HASH_MISMATCH" : "MISSING")},
                        {"expected_sha256", hash},
                        {"actual_sha256",
                         is_error(actual) ? Json(nullptr) : Json(std::get<std::string>(actual))}};
            report["artifact_checks"].push_back(row);
            if (valid)
                roles.insert(role);
            failed |= exists && !valid;
        }
        std::set<std::string> pair_ids, failed_gates;
        for (const auto &pair : manifest.at("tensor_pairs")) {
            exact(pair, {"id", "gate_id", "reference", "candidate", "shape", "dtype"},
                  "tensor_pairs");
            const auto id = string(pair.at("id"), "id"),
                       gate = string(pair.at("gate_id"), "gate_id");
            if (!pair_ids.insert(id).second || !gate_ids.contains(gate) ||
                pair.at("dtype") != "f32le")
                return Error{"EVIDENCE_SCHEMA", id,
                             "Duplicate pair id, unknown gate, or unsupported dtype (use f32le)"};
            const auto &shape = pair.at("shape");
            if (!shape.is_array() || shape.empty() || shape.size() > 8)
                return Error{"TENSOR_SHAPE", id, "Expected 1..8 positive integer axes"};
            std::uint64_t count = 1;
            for (const auto &axis : shape) {
                if (!axis.is_number_unsigned() || axis.get<std::uint64_t>() == 0 ||
                    axis.get<std::uint64_t>() > 1000000000 / count)
                    return Error{"TENSOR_SHAPE", id, "Shape must contain 1..1000000000 elements"};
                count *= axis.get<std::uint64_t>();
            }
            const auto ref_id = string(pair.at("reference"), "reference"),
                       cand_id = string(pair.at("candidate"), "candidate");
            if (!artifacts.contains(ref_id) || !artifacts.contains(cand_id))
                return Error{"ARTIFACT_REFERENCE", id,
                             "Tensor pair references an unknown artifact"};
            Json result = {{"id", id},
                           {"gate_id", gate},
                           {"shape", shape},
                           {"dtype", "f32le"},
                           {"diagnostic_status", "BLOCKED"},
                           {"certifies_model", false}};
            if (artifacts.at(ref_id).valid && artifacts.at(cand_id).valid) {
                result.update(compare(artifacts.at(ref_id), artifacts.at(cand_id), count));
                if (result.at("diagnostic_status") == "FAIL") {
                    failed = true;
                    failed_gates.insert(gate);
                }
            }
            report["tensor_checks"].push_back(result);
        }
        for (auto &gate : report["gates"]) {
            Json missing = Json::array();
            for (const auto &role : gate.at("required_artifact_roles"))
                if (!roles.contains(role.get<std::string>()))
                    missing.push_back(role);
            gate["missing_artifact_roles"] = missing;
            gate["status"] =
                failed_gates.contains(gate.at("id").get<std::string>())
                    ? "FAILED"
                    : (missing.empty() ? "EVIDENCE_PRESENT_UNREVIEWED" : "MISSING_EVIDENCE");
        }
        // Issuance has no code path in this build: calibrated full-model gates do not exist yet.
        report["status"] = failed ? "FAILED" : "BLOCKED";
        return report.dump(2);
    } catch (const Error &error) {
        return error;
    } catch (const Json::exception &error) {
        return Error{"INVALID_JSON", "manifest", error.what()};
    } catch (const std::filesystem::filesystem_error &) {
        return Error{"FILE_READ", "bundle", "Cannot access evidence bundle"};
    }
}
} // namespace seedvr2
