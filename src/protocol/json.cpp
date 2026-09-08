#include "seedvr2/protocol.hpp"
#include <initializer_list>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>

namespace seedvr2 {
namespace {
using Json = nlohmann::json;
void exact_keys(const Json &value, std::initializer_list<std::string_view> keys,
                const std::string &path) {
    if (!value.is_object())
        throw Error{"INVALID_TYPE", path, "Expected an object"};
    for (const auto &key : keys)
        if (!value.contains(std::string(key)))
            throw Error{"MISSING_FIELD", path + "/" + std::string(key),
                        "Required field is missing"};
    for (const auto &item : value.items()) {
        bool found = false;
        for (const auto &key : keys)
            found = found || item.key() == key;
        if (!found)
            throw Error{"UNKNOWN_FIELD", path + "/" + item.key(),
                        "Unknown fields are rejected in schema 1.0"};
    }
}
std::string read_string(const Json &value, const std::string &path) {
    if (!value.is_string())
        throw Error{"INVALID_TYPE", path, "Expected a string"};
    return value.get<std::string>();
}
std::uint32_t read_uint(const Json &value, const std::string &path) {
    if (!value.is_number_integer())
        throw Error{"INVALID_TYPE", path, "Expected a positive integer, not a float or boolean"};
    if (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)
        throw Error{"INVALID_INTEGER", path, "Negative integers are not allowed"};
    const auto number = value.get<std::uint64_t>();
    if (number == 0 || number > std::numeric_limits<std::uint32_t>::max())
        throw Error{"INVALID_INTEGER", path, "Integer must be in 1..4294967295"};
    return static_cast<std::uint32_t>(number);
}
Json extent_json(Extent e) {
    return {{"width", e.width}, {"height", e.height}, {"frames", e.frames}};
}
Json grid_json(TokenGrid grid) { return Json::array({grid.t, grid.h, grid.w}); }
Json window_json(const WindowPlan &plan, bool detailed) {
    Json result = {{"mode", plan.shifted ? "shifted" : "regular"},
                   {"token_grid_thw", grid_json(plan.grid)},
                   {"nominal_window_thw", grid_json(plan.nominal_window)},
                   {"window_count", plan.windows.size()},
                   {"order", "outer-w-middle-h-inner-t"},
                   {"bounds", "half-open"},
                   {"semantics", "seedvr2-3b-720p-window-v1"}};
    if (detailed) {
        result["windows"] = Json::array();
        for (const auto &w : plan.windows)
            result["windows"].push_back({{"t", {w.t.begin, w.t.end}},
                                         {"h", {w.h.begin, w.h.end}},
                                         {"w", {w.w.begin, w.w.end}},
                                         {"packed_offset", w.packed_offset},
                                         {"video_tokens", w.token_count}});
    }
    return result;
}
} // namespace

Result<PlanningRequest> parse_planning_request(std::string_view text) {
    if (text.size() > max_request_bytes)
        return Error{"REQUEST_TOO_LARGE", "", "Planning request exceeds 1 MiB"};
    try {
        std::vector<std::set<std::string>> object_keys;
        auto callback = [&](int depth, Json::parse_event_t event, Json &parsed) {
            if (depth > 32)
                throw Error{"JSON_DEPTH", "", "JSON nesting exceeds 32 levels"};
            if (event == Json::parse_event_t::object_start)
                object_keys.emplace_back();
            else if (event == Json::parse_event_t::object_end)
                object_keys.pop_back();
            else if (event == Json::parse_event_t::key) {
                const auto key = parsed.get<std::string>();
                if (!object_keys.back().insert(key).second)
                    throw Error{"DUPLICATE_KEY", key, "Duplicate JSON keys are not accepted"};
            }
            return true;
        };
        const auto root = Json::parse(text, callback);
        if (!root.is_object())
            return Error{"INVALID_TYPE", "", "Expected an object"};
        if (!root.contains("schema_version"))
            return Error{"MISSING_FIELD", "/schema_version", "Required field is missing"};
        if (read_string(root.at("schema_version"), "/schema_version") != "1.0")
            return Error{"SCHEMA_UNSUPPORTED", "/schema_version",
                         "Only planning-request schema 1.0 is implemented"};
        exact_keys(root, {"schema_version", "model_id", "geometry_policy", "media", "output"}, "");
        if (read_string(root.at("model_id"), "/model_id") != "seedvr2-3b")
            return Error{"MODEL_UNSUPPORTED", "/model_id",
                         "This planner describes SeedVR2-3B only"};
        if (read_string(root.at("geometry_policy"), "/geometry_policy") != "preserve-content-v1")
            return Error{"GEOMETRY_UNSUPPORTED", "/geometry_policy",
                         "Only preserve-content-v1 planning is implemented"};
        const auto &media = root.at("media");
        const auto &output = root.at("output");
        exact_keys(media, {"kind", "width", "height", "frames"}, "/media");
        exact_keys(output, {"scale"}, "/output");
        const auto &scale = output.at("scale");
        exact_keys(scale, {"numerator", "denominator"}, "/output/scale");
        PlanningRequest request;
        const auto kind = read_string(media.at("kind"), "/media/kind");
        if (kind != "image" && kind != "video")
            return Error{"INVALID_KIND", "/media/kind", "Expected image or video"};
        request.kind = kind == "image" ? MediaKind::image : MediaKind::video;
        request.input = {read_uint(media.at("width"), "/media/width"),
                         read_uint(media.at("height"), "/media/height"),
                         read_uint(media.at("frames"), "/media/frames")};
        request.scale = {read_uint(scale.at("numerator"), "/output/scale/numerator"),
                         read_uint(scale.at("denominator"), "/output/scale/denominator")};
        const auto checked = plan_geometry(request);
        if (is_error(checked))
            return std::get<Error>(checked);
        return request;
    } catch (const Error &error) {
        return error;
    } catch (const Json::exception &error) {
        return Error{"INVALID_JSON", "", error.what()};
    }
}

Result<std::string> evaluate_planning_request(std::string_view text) {
    const auto parsed = parse_planning_request(text);
    if (is_error(parsed))
        return std::get<Error>(parsed);
    const auto &request = std::get<PlanningRequest>(parsed);
    const auto geometry = plan_geometry(request);
    if (is_error(geometry))
        return std::get<Error>(geometry);
    const auto &planned = std::get<GeometryPlan>(geometry);
    const auto regular = plan_windows(planned.tokens, false);
    if (is_error(regular))
        return std::get<Error>(regular);
    const auto shifted = plan_windows(planned.tokens, true);
    if (is_error(shifted))
        return std::get<Error>(shifted);
    return serialize_plan(request, planned, std::get<WindowPlan>(regular),
                          std::get<WindowPlan>(shifted));
}

std::string serialize_plan(const PlanningRequest &request, const GeometryPlan &geometry,
                           const WindowPlan &regular, const WindowPlan &shifted) {
    return Json{
        {"schema_version", "1.0"},
        {"document_type", "geometry-plan"},
        {"planning_status", "GEOMETRY_VALID"},
        {"execution_status", "NOT_RUN"},
        {"runnable", false},
        {"model_id", "seedvr2-3b"},
        {"geometry_policy", "preserve-content-v1"},
        {"input_metadata_source", "declared-not-probed"},
        {"declared_input", extent_json(request.input)},
        {"scale",
         {{"numerator", request.scale.numerator}, {"denominator", request.scale.denominator}}},
        {"logical_output", extent_json(geometry.logical)},
        {"working_extent", extent_json(geometry.working)},
        {"padding",
         {{"right", geometry.working.width - geometry.logical.width},
          {"bottom", geometry.working.height - geometry.logical.height},
          {"tail_frames", geometry.working.frames - geometry.logical.frames},
          {"fill", "repeat-edge"},
          {"applied", false}}},
        {"token_grid_thw", grid_json(geometry.tokens)},
        {"video_tokens", geometry.token_count},
        {"windows", {window_json(regular, false), window_json(shifted, false)}},
        {"memory",
         {{"single_hidden_fp16_bytes", geometry.hidden_fp16_bytes},
          {"kind", "theoretical-single-buffer"},
          {"total_peak_bytes", nullptr},
          {"reason",
           "Weights, VAE, temporaries, staging and device allocations are not measured"}}},
        {"execution_blockers",
         {"ENGINE_NOT_BUILT", "INPUT_NOT_PROBED", "MODEL_NOT_VERIFIED", "DEVICE_NOT_PROBED"}}}
        .dump(2);
}
std::string serialize_windows(const WindowPlan &plan) {
    auto result = window_json(plan, true);
    result["schema_version"] = "1.0";
    result["document_type"] = "window-plan";
    result["attention_execution_status"] = "NOT_RUN";
    return result.dump(2);
}
std::string serialize_error(const Error &error) {
    return Json{
        {"schema_version", "1.0"},
        {"document_type", "error"},
        {"error", {{"code", error.code}, {"field", error.field}, {"message", error.message}}}}
        .dump(2);
}
} // namespace seedvr2
