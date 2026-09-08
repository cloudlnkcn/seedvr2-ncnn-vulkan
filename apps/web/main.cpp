#include "seedvr2/protocol.hpp"
#include "web_assets.hpp"
#include <array>
#include <charconv>
#include <httplib.h>
#include <iostream>
#include <stdexcept>
#if defined(_WIN32)
#include <bcrypt.h>
#elif defined(__linux__)
#include <cerrno>
#include <sys/random.h>
#else
#error "Local Web host currently targets Windows and Linux only"
#endif

namespace {
using namespace seedvr2;

std::string session_token() {
    std::array<unsigned char, 32> bytes{};
#if defined(_WIN32)
    if (BCryptGenRandom(nullptr, bytes.data(), static_cast<ULONG>(bytes.size()),
                        BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0)
        throw std::runtime_error("Cannot initialize local session");
#else
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto count = getrandom(bytes.data() + offset, bytes.size() - offset, 0);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            throw std::runtime_error("Cannot initialize local session");
        offset += static_cast<std::size_t>(count);
    }
#endif
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : bytes) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}

void fail(httplib::Response &response, int status, const Error &error) {
    response.status = status;
    response.set_content(serialize_error(error), "application/json; charset=utf-8");
}

void asset(httplib::Response &response, std::string_view text, const char *mime) {
    response.set_content(text.data(), text.size(), mime);
}

int serve(int port) {
    const auto token = session_token();
    httplib::Server server;
    // Four bounded I/O workers. Model execution must use the separate worker process.
    server.new_task_queue = [] { return new httplib::ThreadPool(4, 4, 16); };
    server.set_payload_max_length(max_request_bytes);
    server.set_read_timeout(5);
    server.set_write_timeout(10);
    server.set_keep_alive_timeout(2);
    server.set_keep_alive_max_count(32);
    server.set_default_headers(
        {{"Cache-Control", "no-store"},
         {"X-Content-Type-Options", "nosniff"},
         {"Referrer-Policy", "no-referrer"},
         {"Cross-Origin-Resource-Policy", "same-origin"},
         {"X-Frame-Options", "DENY"},
         {"Content-Security-Policy",
          "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
          "img-src 'self' blob:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"}});

    if (port == 0)
        port = server.bind_to_any_port("127.0.0.1");
    else if (!server.bind_to_port("127.0.0.1", port))
        port = -1;
    if (port <= 0) {
        std::cerr << serialize_error({"PORT_UNAVAILABLE", "port",
                                      "Cannot bind loopback port; choose --port 0 or another port"})
                  << '\n';
        return 3;
    }
    const auto authority = "127.0.0.1:" + std::to_string(port);
    const auto origin = "http://" + authority;
    // Check before reading request bodies. The host is deliberately not a LAN service.
    server.set_pre_request_handler([&](const auto &request, auto &response) {
        if (request.get_header_value_count("Host") != 1 ||
            request.get_header_value("Host") != authority) {
            fail(response, 403, {"HOST_REJECTED", "Host", "Use the exact local launch address"});
            return httplib::Server::HandlerResponse::Handled;
        }
        const auto fetch_site = request.get_header_value("Sec-Fetch-Site");
        // A link may open the home page from another site. API/asset reads still require
        // same-origin access; no embedded page or cross-origin script gets a session.
        const bool home_navigation = request.method == "GET" && request.path == "/" &&
                                     request.get_header_value("Sec-Fetch-Mode") == "navigate" &&
                                     request.get_header_value("Sec-Fetch-Dest") == "document";
        if (request.get_header_value_count("Origin") > 1 ||
            (request.has_header("Origin") && request.get_header_value("Origin") != origin) ||
            (!home_navigation && !fetch_site.empty() && fetch_site != "same-origin" &&
             fetch_site != "none")) {
            fail(response, 403, {"ORIGIN_REJECTED", "Origin", "Same-origin local access required"});
            return httplib::Server::HandlerResponse::Handled;
        }
        if (request.method != "GET" && request.method != "HEAD" &&
            (request.get_header_value_count("X-SeedVR2-Session") != 1 ||
             request.get_header_value("X-SeedVR2-Session") != token)) {
            fail(response, 403, {"SESSION_REQUIRED", "session", "Reload the local application"});
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    server.set_error_handler([](const auto &, auto &response) {
        if (!response.body.empty())
            return;
        if (response.status == 413)
            fail(response, 413, {"REQUEST_TOO_LARGE", "request", "Planning request exceeds 1 MiB"});
        else
            fail(response, response.status,
                 {"HTTP_ERROR", "request", "No matching local endpoint"});
    });
    server.set_exception_handler([](const auto &, auto &response, std::exception_ptr) {
        fail(response, 500, {"INTERNAL_ERROR", "", "Local request failed"});
    });
    server.Get("/", [](const auto &, auto &response) {
        asset(response, web_assets::html, "text/html; charset=utf-8");
    });
    server.Get("/app.css", [](const auto &, auto &response) {
        asset(response, web_assets::css, "text/css; charset=utf-8");
    });
    server.Get("/app.js", [](const auto &, auto &response) {
        asset(response, web_assets::js, "text/javascript; charset=utf-8");
    });
    server.Get("/favicon.ico", [](const auto &, auto &response) { response.status = 204; });
    server.Get("/api/v1/session", [&](const auto &, auto &response) {
        response.set_content("{\"schema_version\":\"1.0\",\"session_token\":\"" + token + "\"}",
                             "application/json; charset=utf-8");
    });
    server.Get("/api/v1/capabilities", [](const auto &, auto &response) {
        response.set_content(
            R"({"schema_version":"1.0","build":"0.1.0-framework+local-web","frontend":"local-web","planning":true,"inference":false,"gpu_probe":false,"media_import":false,"persistent_queue":false,"events":false,"model_validation":"NOT_RUN","bind":"loopback-only"})",
            "application/json; charset=utf-8");
    });
    server.Post("/api/v1/plan", [](const auto &request, auto &response) {
        if (request.get_header_value_count("Content-Type") != 1 ||
            request.get_header_value("Content-Type") != "application/json") {
            fail(response, 415, {"CONTENT_TYPE", "Content-Type", "Use application/json"});
            return;
        }
        const auto result = evaluate_planning_request(request.body);
        if (is_error(result)) {
            fail(response, 422, std::get<Error>(result));
            return;
        }
        response.set_content(std::get<std::string>(result), "application/json; charset=utf-8");
    });
    server.set_start_handler([&] {
        // Tokens never appear in the URL or startup log.
        std::cout << "{\"event\":\"listening\",\"url\":\"" << origin << "/\",\"inference\":false}"
                  << std::endl;
    });
    return server.listen_after_bind() ? 0 : 3;
}
} // namespace

int main(int argc, char **argv) {
    int port = 8877;
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "SeedVR2 local Web framework (planning only)\n"
                     "  seedvr2-web [--port <0..65535>]\n"
                     "Open the printed loopback URL. --port 0 selects a free port.\n";
        return 0;
    }
    if (argc != 1) {
        if (argc != 3 || std::string_view(argv[1]) != "--port")
            return 2;
        const std::string_view value = argv[2];
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), port);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || port < 0 ||
            port > 65535)
            return 2;
    }
    try {
        return serve(port);
    } catch (const std::exception &error) {
        std::cerr << serialize_error({"WEB_START_FAILED", "", error.what()}) << '\n';
        return 5;
    }
}
