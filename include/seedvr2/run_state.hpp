#pragma once
#include "seedvr2/result.hpp"
#include <cstdint>
#include <string>
#include <string_view>

namespace seedvr2 {
enum class RunState {
    created,
    validating,
    queued,
    running,
    cancelling,
    committing,
    succeeded,
    rejected,
    cancelled,
    failed,
    interrupted
};
struct RunSnapshot {
    std::string run_id;
    RunState state = RunState::created;
    std::uint64_t last_revision = 0;
};
struct TransitionEvent {
    std::string run_id;
    // Counts state changes only; transport progress events use a separate event_seq.
    std::uint64_t revision = 0;
    RunState next = RunState::created;
};
[[nodiscard]] bool is_terminal(RunState state);
[[nodiscard]] std::string_view to_string(RunState state);
[[nodiscard]] Result<RunSnapshot> apply_transition(const RunSnapshot &current,
                                                   const TransitionEvent &event);
} // namespace seedvr2
