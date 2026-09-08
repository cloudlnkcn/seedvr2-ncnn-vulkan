#pragma once

#include <string>
#include <variant>

namespace seedvr2 {
struct Error {
    std::string code;
    std::string field;
    std::string message;
};

template <class T> using Result = std::variant<T, Error>;
template <class T> bool is_error(const Result<T> &result) {
    return std::holds_alternative<Error>(result);
}
} // namespace seedvr2
