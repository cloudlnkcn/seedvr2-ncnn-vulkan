#include "seedvr2/run_state.hpp"
#include <limits>

namespace seedvr2 {
bool is_terminal(RunState state) {
    return state == RunState::succeeded || state == RunState::rejected ||
           state == RunState::cancelled || state == RunState::failed ||
           state == RunState::interrupted;
}
std::string_view to_string(RunState state) {
    switch (state) {
    case RunState::created:
        return "created";
    case RunState::validating:
        return "validating";
    case RunState::queued:
        return "queued";
    case RunState::running:
        return "running";
    case RunState::cancelling:
        return "cancelling";
    case RunState::committing:
        return "committing";
    case RunState::succeeded:
        return "succeeded";
    case RunState::rejected:
        return "rejected";
    case RunState::cancelled:
        return "cancelled";
    case RunState::failed:
        return "failed";
    case RunState::interrupted:
        return "interrupted";
    }
    return "invalid";
}
Result<RunSnapshot> apply_transition(const RunSnapshot &current, const TransitionEvent &event) {
    if (current.run_id.empty() || event.run_id != current.run_id)
        return Error{"RUN_ID_MISMATCH", "run_id", "Event belongs to a different or empty run"};
    if (current.last_revision == std::numeric_limits<std::uint64_t>::max() ||
        event.revision != current.last_revision + 1)
        return Error{"STATE_REVISION", "revision",
                     "Transition revision must be contiguous and unique"};
    if (is_terminal(current.state))
        return Error{"TERMINAL_RUN", "state",
                     "Retry creates a new run; completed runs are immutable"};
    const auto next = event.next;
    bool allowed = false;
    switch (current.state) {
    case RunState::created:
        allowed = next == RunState::validating || next == RunState::cancelled;
        break;
    case RunState::validating:
        allowed = next == RunState::queued || next == RunState::rejected ||
                  next == RunState::cancelled || next == RunState::interrupted;
        break;
    case RunState::queued:
        allowed = next == RunState::running || next == RunState::cancelled;
        break;
    case RunState::running:
        allowed = next == RunState::committing || next == RunState::cancelling ||
                  next == RunState::failed || next == RunState::interrupted;
        break;
    case RunState::cancelling:
        allowed = next == RunState::cancelled || next == RunState::failed ||
                  next == RunState::interrupted;
        break;
    case RunState::committing:
        allowed = next == RunState::succeeded || next == RunState::failed ||
                  next == RunState::interrupted;
        break;
    default:
        break;
    }
    if (!allowed)
        return Error{"INVALID_TRANSITION", "state", "Requested run transition is not allowed"};
    return RunSnapshot{current.run_id, next, event.revision};
}
} // namespace seedvr2
