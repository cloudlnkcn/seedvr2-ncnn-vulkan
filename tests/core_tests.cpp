#include "seedvr2/planning.hpp"
#include "seedvr2/protocol.hpp"
#include "seedvr2/run_state.hpp"
#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

using namespace seedvr2;
using Json = nlohmann::json;
namespace {
int assertions = 0;
#define CHECK(...)                                                                                 \
    do {                                                                                           \
        ++assertions;                                                                              \
        if (!(__VA_ARGS__))                                                                        \
            throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " +      \
                                     #__VA_ARGS__);                                                \
    } while (false)
template <class T> T take(Result<T> result) {
    if (is_error(result))
        throw std::runtime_error(std::get<Error>(result).code);
    return std::get<T>(std::move(result));
}
PlanningRequest video(std::uint32_t width, std::uint32_t height, std::uint32_t frames) {
    return {MediaKind::video, {width, height, frames}, {1, 1}};
}
Json request_json() {
    return {{"schema_version", "1.0"},
            {"model_id", "seedvr2-3b"},
            {"geometry_policy", "preserve-content-v1"},
            {"media", {{"kind", "video"}, {"width", 640}, {"height", 360}, {"frames", 17}}},
            {"output", {{"scale", {{"numerator", 2}, {"denominator", 1}}}}}};
}
void geometry_cases() {
    const auto p = take(plan_geometry({MediaKind::video, {640, 360, 17}, {2, 1}}));
    CHECK(p.tokens == TokenGrid{5, 45, 80});
    CHECK(p.logical == Extent{1280, 720, 17});
    CHECK(p.working == p.logical);
    CHECK(p.token_count == 18000);
    CHECK(p.hidden_fp16_bytes == 92160000);
    const auto full = take(plan_geometry(video(1920, 1080, 16)));
    CHECK(full.logical == Extent{1920, 1080, 16});
    CHECK(full.working == Extent{1920, 1088, 17});
    CHECK(full.tokens == TokenGrid{5, 68, 120});
    CHECK(take(plan_geometry(video(1, 1, 1))).working == Extent{16, 16, 1});
    CHECK(take(plan_geometry({MediaKind::image, {7, 5, 1}, {1, 2}})).logical == Extent{4, 3, 1});
    for (const auto f : {1U, 2U, 4U, 5U, 16U, 17U, 18U, 33U, 4097U}) {
        const auto temporal = take(plan_geometry(video(16, 16, f)));
        CHECK((temporal.working.frames - 1) % 4 == 0);
        CHECK(temporal.working.frames >= f && temporal.working.frames - f <= 3);
    }
}
void reject_unsafe_geometry() {
    CHECK(is_error(plan_geometry(video(0, 720, 17))));
    CHECK(is_error(plan_geometry(video(1280, 0, 17))));
    CHECK(is_error(plan_geometry(video(1280, 720, 0))));
    CHECK(is_error(plan_geometry({MediaKind::image, {640, 360, 2}, {1, 1}})));
    CHECK(is_error(plan_geometry({MediaKind::image, {640, 360, 1}, {1, 0}})));
    CHECK(is_error(plan_geometry({MediaKind::image, {1, 1, 1}, {1, 64}})));
    CHECK(is_error(plan_geometry({MediaKind::image, {65536, 65536, 1}, {64, 1}})));
    CHECK(is_error(plan_geometry(video(65536, 65536, 4097))));
    CHECK(is_error(plan_geometry(video(4294967295U, 4294967295U, 4294967295U))));
    CHECK(is_error(plan_windows({4294967295U, 4294967295U, 4294967295U}, false)));
    CHECK(is_error(plan_windows({100000000, 1, 1}, false)));
    CHECK(is_error(plan_windows({0, 1, 1}, false)));
    const auto aspect = plan_windows({1, 1, 14400}, false);
    CHECK(is_error(aspect) && std::get<Error>(aspect).code == "UNSUPPORTED_ASPECT");
}
void reference_windows(const std::string &path) {
    std::ifstream stream(path);
    CHECK(stream.good());
    const auto reference = Json::parse(stream);
    CHECK(reference.at("cases").size() == 16);
    for (const auto &sample : reference.at("cases")) {
        const auto axes = sample.at("grid").get<std::array<std::uint32_t, 3>>();
        const TokenGrid grid{axes[0], axes[1], axes[2]};
        const auto plan = take(plan_windows(grid, sample.at("shifted").get<bool>()));
        CHECK(plan.windows.size() == sample.at("windows").size());
        std::vector<std::uint8_t> coverage(std::size_t{grid.t} * grid.h * grid.w, 0);
        std::uint32_t offset = 0;
        for (std::size_t i = 0; i < plan.windows.size(); ++i) {
            const auto &window = plan.windows[i];
            const auto expected = sample.at("windows").at(i).get<std::array<std::uint32_t, 6>>();
            CHECK(window.t == Interval{expected[0], expected[1]});
            CHECK(window.h == Interval{expected[2], expected[3]});
            CHECK(window.w == Interval{expected[4], expected[5]});
            CHECK(window.packed_offset == offset);
            CHECK(window.token_count == (window.t.end - window.t.begin) *
                                            (window.h.end - window.h.begin) *
                                            (window.w.end - window.w.begin));
            offset += window.token_count;
            for (auto t = window.t.begin; t < window.t.end; ++t)
                for (auto h = window.h.begin; h < window.h.end; ++h)
                    for (auto w = window.w.begin; w < window.w.end; ++w) {
                        const auto index = (std::size_t{t} * grid.h + h) * grid.w + w;
                        CHECK(index < coverage.size());
                        CHECK(coverage[index] == 0);
                        coverage[index] = 1;
                    }
        }
        CHECK(offset == coverage.size());
        for (const auto covered : coverage)
            CHECK(covered == 1);
    }
    // 60*sqrt(25/576) == 12.5. Python round(12.5) is 12, not 13.
    CHECK(take(plan_windows({1, 25, 576}, false)).nominal_window.h == 4);
}
void transition_matrix() {
    using S = RunState;
    const std::set<std::pair<S, S>> allowed{
        {S::created, S::validating},     {S::created, S::cancelled},
        {S::validating, S::queued},      {S::validating, S::rejected},
        {S::validating, S::cancelled},   {S::validating, S::interrupted},
        {S::queued, S::running},         {S::queued, S::cancelled},
        {S::running, S::committing},     {S::running, S::cancelling},
        {S::running, S::failed},         {S::running, S::interrupted},
        {S::cancelling, S::cancelled},   {S::cancelling, S::failed},
        {S::cancelling, S::interrupted}, {S::committing, S::succeeded},
        {S::committing, S::failed},      {S::committing, S::interrupted}};
    const std::array states{S::created,    S::validating, S::queued,     S::running,
                            S::cancelling, S::committing, S::succeeded,  S::rejected,
                            S::cancelled,  S::failed,     S::interrupted};
    for (const auto from : states)
        for (const auto to : states) {
            const auto result = apply_transition({"run-1", from, 4}, {"run-1", 5, to});
            CHECK(!is_error(result) == allowed.contains({from, to}));
        }
    RunSnapshot run{"run-1"};
    std::uint64_t seq = 0;
    for (const auto next : {S::validating, S::queued, S::running, S::cancelling, S::cancelled}) {
        run = take(apply_transition(run, {"run-1", ++seq, next}));
        CHECK(run.state == next && run.last_revision == seq);
    }
    CHECK(is_terminal(run.state));
    CHECK(is_error(apply_transition(run, {"run-1", ++seq, S::queued})));
}
void reject_stale_events() {
    using S = RunState;
    const RunSnapshot run{"run-1", S::running, 3};
    CHECK(is_error(apply_transition(run, {"run-2", 4, S::failed})));
    CHECK(is_error(apply_transition(run, {"run-1", 3, S::failed})));
    CHECK(is_error(apply_transition(run, {"run-1", 5, S::failed})));
    CHECK(is_error(apply_transition({"", S::running, 3}, {"", 4, S::failed})));
    CHECK(is_error(apply_transition({"run-1", S::running, UINT64_MAX}, {"run-1", 0, S::failed})));
    CHECK(run.last_revision == 3 && run.state == S::running);
}
void protocol_rejections() {
    const auto good = request_json();
    CHECK(!is_error(parse_planning_request(good.dump())));
    for (const auto &wrong : {Json(-1), Json(0), Json(17.0), Json(true), Json(nullptr), Json("17"),
                              Json(UINT64_MAX), Json(4098)}) {
        auto candidate = good;
        candidate["media"]["frames"] = wrong;
        CHECK(is_error(parse_planning_request(candidate.dump())));
    }
    for (const auto &field : {"schema_version", "model_id", "geometry_policy"}) {
        auto candidate = good;
        candidate[field] = "unrecognized";
        CHECK(is_error(parse_planning_request(candidate.dump())));
    }
    for (const auto &field : {"schema_version", "media", "output"}) {
        auto candidate = good;
        candidate.erase(field);
        CHECK(is_error(parse_planning_request(candidate.dump())));
    }
    for (const auto &path : {"", "/media", "/output", "/output/scale"}) {
        auto candidate = good;
        candidate[Json::json_pointer(path)]["typo"] = true;
        const auto result = parse_planning_request(candidate.dump());
        CHECK(is_error(result) && std::get<Error>(result).code == "UNKNOWN_FIELD");
    }
    const auto duplicate = parse_planning_request(R"({"media":{"frames":1,"frames":2}})");
    CHECK(is_error(duplicate) && std::get<Error>(duplicate).code == "DUPLICATE_KEY");
    CHECK(is_error(parse_planning_request("[]")));
    CHECK(is_error(parse_planning_request("null")));
    CHECK(is_error(parse_planning_request("{")));
    const auto large = parse_planning_request(std::string(max_request_bytes + 1, ' '));
    CHECK(is_error(large) && std::get<Error>(large).code == "REQUEST_TOO_LARGE");
    const auto deep = parse_planning_request(std::string(40, '[') + "0" + std::string(40, ']'));
    CHECK(is_error(deep) && std::get<Error>(deep).code == "JSON_DEPTH");
}
void honest_plan_output() {
    const auto request = take(parse_planning_request(request_json().dump()));
    const auto geometry = take(plan_geometry(request));
    const auto regular = take(plan_windows(geometry.tokens, false));
    const auto shifted = take(plan_windows(geometry.tokens, true));
    const auto output = Json::parse(serialize_plan(request, geometry, regular, shifted));
    CHECK(output.at("runnable") == false);
    CHECK(output.at("execution_status") == "NOT_RUN");
    CHECK(output.at("memory").at("total_peak_bytes").is_null());
    CHECK(output.at("windows").at(0).at("window_count") == 27);
    CHECK(output.at("windows").at(1).at("window_count") == 48);
    CHECK(output.at("execution_blockers").size() == 4);
    CHECK(Json::parse(serialize_error({"CODE", "/field", "quoted \"message\""}))
              .at("error")
              .at("code") == "CODE");
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    int failures = 0;
    const auto run = [&](const char *name, const std::function<void()> &test) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception &error) {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    };
    run("geometry and temporal padding", geometry_cases);
    run("unsafe geometry and allocation limits", reject_unsafe_geometry);
    run("ordered official window reference and single coverage",
        [&] { reference_windows(argv[1]); });
    run("run transition matrix and cancellation", transition_matrix);
    run("stale and foreign events", reject_stale_events);
    run("strict JSON boundary", protocol_rejections);
    run("plan cannot claim inference or measured VRAM", honest_plan_output);
    std::cout << "groups=7 failures=" << failures << " assertions=" << assertions << '\n';
    return failures == 0 ? 0 : 1;
}
