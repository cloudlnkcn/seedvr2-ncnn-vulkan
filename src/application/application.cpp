#include "seedvr2/application.hpp"
#include "seedvr2/engine.hpp"
#include "seedvr2/protocol.hpp"
#include "seedvr2/validation.hpp"
#include <nlohmann/json.hpp>

namespace seedvr2 {
using Json = nlohmann::json;
Result<std::string> Application::save_plan(std::string_view request) {
    const auto plan = evaluate_planning_request(request);
    if (is_error(plan))
        return std::get<Error>(plan);
    const auto payload = Json{{"schema_version", "1.0"},
                              {"document_type", "saved-plan"},
                              {"request", Json::parse(request)},
                              {"plan", Json::parse(std::get<std::string>(plan))},
                              {"execution_status", "NOT_RUN"}}
                             .dump();
    return workspace_.save("plan", "PLANNED", payload);
}
Result<std::string> Application::audit_and_save(const std::filesystem::path &bundle) {
    const auto audit = audit_model_bundle(bundle);
    if (is_error(audit))
        return std::get<Error>(audit);
    auto payload = Json::parse(std::get<std::string>(audit));
    const auto saved =
        workspace_.save("model-audit", payload.at("status").get<std::string>(), payload.dump());
    if (is_error(saved))
        return std::get<Error>(saved);
    payload["record_id"] = Json::parse(std::get<std::string>(saved)).at("id");
    return payload.dump(2);
}
Result<std::string> Application::self_test_and_save(std::string_view request) {
    try {
        int keys = 0;
        const auto doc = Json::parse(request, [&](int depth, Json::parse_event_t event, Json &) {
            if (depth > 1)
                throw std::runtime_error("Self-test request must be a flat object");
            if (event == Json::parse_event_t::key)
                ++keys;
            return true;
        });
        if (!doc.is_object() || doc.size() != 2 || keys != 2 || !doc.contains("backend") ||
            !doc.contains("gpu") || (doc.at("backend") != "cpu" && doc.at("backend") != "vulkan") ||
            !doc.at("gpu").is_number_integer() || doc.at("gpu") < -1 || doc.at("gpu") > 64)
            throw std::runtime_error("Expected backend cpu/vulkan and integer gpu -1..64");
        const auto result = engine::self_test(doc.at("backend") == "vulkan", doc.at("gpu").get<int>());
        if (is_error(result))
            return std::get<Error>(result);
        auto payload = Json::parse(std::get<std::string>(result));
        const auto saved = workspace_.save("operator-test", payload.at("status").get<std::string>(), payload.dump());
        if (is_error(saved))
            return std::get<Error>(saved);
        payload["record_id"] = Json::parse(std::get<std::string>(saved)).at("id");
        return payload.dump(2);
    } catch (const std::exception &e) {
        return Error{"SELF_TEST_REQUEST", "request", e.what()};
    }
}
std::string application_capabilities() {
    return Json{{"schema_version", "1.0"},
                {"build", "0.7.0-native-preview"},
                {"architecture", "shared-native-sdk"},
                {"installed_cpp_sdk",true},
                {"reviewed_package_identity",true},
                {"offline_model_copy",true},
                {"frontend", "react-typescript-antd"},
                {"http_framework", "Drogon 1.9.13"},
                {"planning", true},
                {"saved_plans", true},
                {"evidence_audit", true},
                {"audit_cli_only", true},
                {"model_certification", false},
                {"inference", true},
                {"ncnn_linked", true},
                {"vulkan_execution", true},
                {"gpu_probe", true},
                {"operator_self_test", true},
                {"execution_scope", "Image and whole-clip video FP32-B pipelines, 32 DiT blocks, ncnn CPU/Vulkan; video up to 17 frames and 128 pixels; no streaming cache"},
                {"engine", Json::parse(engine::build_status())},
                {"media_import", true},
                {"persistent_queue", true},
                {"events", "DURABLE_CURSOR_POLLING"},
                {"model_validation", "CERTIFICATE_NOT_FROZEN"},
                {"bind", "loopback-only"}}
        .dump();
}
} // namespace seedvr2
