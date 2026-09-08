#include "process.hpp"
#include <array>
#include <cerrno>
#include <chrono>
#include <stdexcept>
#if defined(__linux__)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif
namespace seedvr2::jobs_detail {
int run_process(const std::vector<std::string> &arguments, const std::filesystem::path &log,
                const std::function<bool()> &cancelled,
                const std::function<void(const std::string &)> &line) {
#if defined(__linux__)
    int pipefd[2];
    if (pipe2(pipefd, O_CLOEXEC) != 0)
        throw std::runtime_error("Cannot create worker event pipe");
    const int errors = open(log.c_str(), O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC, 0600);
    if (errors < 0) { close(pipefd[0]); close(pipefd[1]); throw std::runtime_error("Cannot create worker log"); }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, errors, STDERR_FILENO);
    posix_spawn_file_actions_addclosefrom_np(&actions, 3);
    std::vector<char *> argv;
    for (const auto &arg : arguments) argv.push_back(const_cast<char *>(arg.c_str()));
    argv.push_back(nullptr);
    pid_t pid = -1;
    const int spawned = posix_spawn(&pid, arguments[0].c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipefd[1]); close(errors);
    if (spawned != 0) { close(pipefd[0]); throw std::runtime_error("Cannot start the isolated inference worker"); }
    fcntl(pipefd[0], F_SETFL, O_NONBLOCK);
    using Clock = std::chrono::steady_clock;
    auto cancelled_at = Clock::time_point{};
    std::string pending;
    std::size_t total = 0;
    bool ended = false, eof = false;
    int status = 0;
    auto terminate = [&] { kill(pid, SIGKILL); while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {} };
    try {
        while (!ended || !eof) {
            if (!ended && cancelled()) {
                if (cancelled_at == Clock::time_point{}) { cancelled_at = Clock::now(); kill(pid, SIGTERM); }
                else if (Clock::now()-cancelled_at > std::chrono::seconds(10)) kill(pid, SIGKILL);
            }
            pollfd p{pipefd[0], POLLIN|POLLHUP, 0};
            const int ready = poll(&p, 1, 100);
            if (ready < 0 && errno != EINTR) throw std::runtime_error("Worker pipe polling failed");
            if (ready > 0) {
                std::array<char, 8192> bytes{};
                for (;;) {
                    const auto size = read(pipefd[0], bytes.data(), bytes.size());
                    if (size == 0) { eof = true; break; }
                    if (size < 0) {
                        if (errno == EAGAIN || errno == EINTR) break;
                        throw std::runtime_error("Worker pipe read failed");
                    }
                    total += static_cast<std::size_t>(size);
                    if (total > 8*1024*1024) throw std::runtime_error("Worker event stream exceeds 8 MiB");
                    pending.append(bytes.data(), static_cast<std::size_t>(size));
                    for (auto newline = pending.find('\n'); newline != std::string::npos; newline = pending.find('\n')) {
                        if (newline > 128*1024) throw std::runtime_error("Oversized worker event");
                        line(pending.substr(0, newline)); pending.erase(0, newline+1);
                    }
                    if (pending.size() > 128*1024) throw std::runtime_error("Oversized worker event");
                }
            }
            if (!ended) {
                const auto reaped = waitpid(pid, &status, WNOHANG);
                if (reaped == pid) ended = true;
                else if (reaped < 0 && errno != EINTR) throw std::runtime_error("Cannot reap worker");
            }
        }
        if (!pending.empty()) throw std::runtime_error("Truncated worker event");
        close(pipefd[0]);
        return WIFEXITED(status) ? WEXITSTATUS(status) : 128+(WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    } catch (...) {
        if (!ended) terminate();
        close(pipefd[0]); throw;
    }
#else
    (void)arguments; (void)log; (void)cancelled; (void)line;
    throw std::runtime_error("The supervised worker adapter is currently available on Linux");
#endif
}
}
