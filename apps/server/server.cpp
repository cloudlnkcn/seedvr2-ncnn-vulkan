#include "server.hpp"
#include "assets.hpp"
#include "seedvr2/application.hpp"
#include "seedvr2/engine.hpp"
#include "seedvr2/jobs.hpp"
#include "seedvr2/protocol.hpp"
#include "seedvr2/validation.hpp"
#include <atomic>
#include <drogon/drogon.h>
#include <iostream>
#include <fstream>
#include <charconv>
#include <openssl/rand.h>
#include <trantor/utils/ConcurrentTaskQueue.h>

namespace seedvr2::server {
namespace {
using namespace drogon;
using Reply = std::function<void(const HttpResponsePtr &)>;
HttpResponsePtr response(std::string body, int code = 200,
                         std::string_view mime = "application/json; charset=utf-8") {
    auto out = HttpResponse::newHttpResponse();
    out->setStatusCode(static_cast<HttpStatusCode>(code));
    out->setContentTypeString(mime);
    out->setBody(std::move(body));
    return out;
}
HttpResponsePtr error(int status, const Error &reason) {
    return response(serialize_error(reason), status);
}
void policy_headers(const HttpResponsePtr &out) {
    out->addHeader("Cache-Control", "no-store");
    out->addHeader("X-Content-Type-Options", "nosniff");
    out->addHeader("Referrer-Policy", "no-referrer");
    out->addHeader("Cross-Origin-Resource-Policy", "same-origin");
    out->addHeader("X-Frame-Options", "DENY");
    // Ant Design uses dynamic styles and inline layout properties. Script execution stays
    // same-origin.
    out->addHeader("Content-Security-Policy",
                   "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                   "connect-src 'self'; img-src 'self' blob: data:; media-src 'self'; font-src 'self'; base-uri "
                   "'none'; frame-ancestors 'none'; form-action 'none'");
}
std::string token() {
    unsigned char bytes[32];
    if (RAND_bytes(bytes, 32) != 1)
        throw std::runtime_error("Cannot initialize local session");
    std::string text;
    constexpr char hex[] = "0123456789abcdef";
    for (auto b : bytes) {
        text += hex[b >> 4];
        text += hex[b & 15];
    }
    return text;
}
struct Services {
    Services(const std::filesystem::path &database, const std::filesystem::path &worker,
             const std::filesystem::path &model, const std::filesystem::path &video_model)
        : application(database), jobs(database, worker, model, video_model), tasks(4, "seedvr2-app") {}
    Application application;
    Jobs jobs;
    const std::string session = token();
    std::atomic<unsigned int> pending{0};
    trantor::ConcurrentTaskQueue tasks;
    void dispatch(Reply callback, std::function<Result<std::string>()> work) {
        if (pending.fetch_add(1) >= 16) {
            --pending;
            callback(error(429, {"BUSY", "request", "Local work queue is full; retry shortly"}));
            return;
        }
        tasks.runTaskInQueue([this, callback = std::move(callback), work = std::move(work)] {
            HttpResponsePtr out;
            try {
                const auto result = work();
                if (is_error(result)) {
                    const auto &why = std::get<Error>(result);
                    out = error(why.code == "RECORD_NOT_FOUND" || why.code == "JOB_NOT_FOUND" ? 404 : 422, why);
                } else
                    out = response(std::get<std::string>(result));
            } catch (...) {
                out = error(500, {"INTERNAL_ERROR", "", "Local operation failed"});
            }
            --pending;
            callback(out);
        });
    }
};
HttpResponsePtr media_response(const std::filesystem::path &path, const HttpRequestPtr &req, const std::string &mime) {
    const auto size=std::filesystem::file_size(path);
    const auto range=req->getHeader("range");
    auto invalid=[&] {
        auto out=error(416,{"RANGE_INVALID","range","Requested media range is unavailable"});
        out->addHeader("Content-Range","bytes */"+std::to_string(size));return out;
    };
    HttpResponsePtr out;
    if (!range.empty()) {
        if (range.size()>80 || !range.starts_with("bytes=") || range.find(',')!=std::string::npos || !size) return invalid();
        const auto value=std::string_view(range).substr(6);const auto dash=value.find('-');
        if (dash==std::string_view::npos) return invalid();
        auto number=[](std::string_view s,std::uint64_t &n) {
            if (s.empty()) return false;
            const auto parsed=std::from_chars(s.data(),s.data()+s.size(),n);
            return parsed.ec==std::errc{} && parsed.ptr==s.data()+s.size();
        };
        std::uint64_t begin=0,end=size-1;
        if (dash==0) {
            std::uint64_t suffix=0;if (!number(value.substr(1),suffix)||!suffix) return invalid();
            begin=size-std::min(size,suffix);
        } else {
            if (!number(value.substr(0,dash),begin)||begin>=size) return invalid();
            if (dash+1<value.size() && !number(value.substr(dash+1),end)) return invalid();
            end=std::min(end,size-1);if (end<begin) return invalid();
        }
        out=HttpResponse::newFileResponse(path.string(),begin,end-begin+1,true,"",CT_NONE,mime,req);
    } else out=HttpResponse::newFileResponse(path.string(),"",CT_NONE,mime,req);
    out->setContentTypeString(mime);out->addHeader("Accept-Ranges","bytes");return out;
}
} // namespace

int serve(int port, const std::filesystem::path &database,
          const std::filesystem::path &worker, const std::filesystem::path &model, const std::filesystem::path &video_model) {
    auto services = std::make_shared<Services>(database, worker, model, video_model);
    auto &host = app();
    host.setLogLevel(trantor::Logger::kWarn)
        .setThreadNum(1)
        .addListener("127.0.0.1", static_cast<std::uint16_t>(port))
        .setClientMaxBodySize(256*1024*1024)
        .setClientMaxMemoryBodySize(256*1024*1024)
        .setMaxConnectionNum(64)
        .setMaxConnectionNumPerIP(64)
        .setIdleConnectionTimeout(10)
        .setKeepaliveRequestsNumber(64)
        .setPipeliningRequestsNumber(8)
        .enableGzip(false)
        .enableBrotli(false)
        .setFileTypes({})
        .disableSession();
    host.registerPreSendingAdvice(
        [](const HttpRequestPtr &, const HttpResponsePtr &out) { policy_headers(out); });
    host.setCustomErrorHandler([](HttpStatusCode code, const HttpRequestPtr &) {
        return error(code, {code == k413RequestEntityTooLarge ? "REQUEST_TOO_LARGE" : "HTTP_ERROR",
                            "request", "Request rejected or no matching endpoint"});
    });
    host.registerPreRoutingAdvice([services](const HttpRequestPtr &req, AdviceCallback &&reject,
                                             AdviceChainCallback &&next) {
        const auto authority = req->getLocalAddr().toIpPort();
        const auto origin = "http://" + authority;
        if (req->getHeader("host") != authority) {
            reject(error(403, {"HOST_REJECTED", "Host", "Use the exact local launch address"}));
            return;
        }
        const auto &site = req->getHeader("sec-fetch-site");
        const bool navigation = req->method() == Get && req->path() == "/" &&
                                req->getHeader("sec-fetch-mode") == "navigate" &&
                                req->getHeader("sec-fetch-dest") == "document";
        if ((req->headers().contains("origin") && req->getHeader("origin") != origin) ||
            (!navigation && !site.empty() && site != "same-origin" && site != "none")) {
            reject(error(403, {"ORIGIN_REJECTED", "Origin", "Same-origin local access required"}));
            return;
        }
        if (req->method() != Get && req->method() != Head &&
            req->getHeader("x-seedvr2-session") != services->session) {
            reject(error(403, {"SESSION_REQUIRED", "session", "Reload the local application"}));
            return;
        }
        const bool upload = req->path() == "/api/v1/media";
        if (!upload && req->body().size() > max_request_bytes) {
            reject(error(413, {"REQUEST_TOO_LARGE", "request", "Control request exceeds 1 MiB"}));
            return;
        }
        if (req->method() == Post && !upload && req->getHeader("content-type") != "application/json") {
            reject(error(415, {"CONTENT_TYPE", "Content-Type", "Use application/json"}));
            return;
        }
        next();
    });
    for (const auto &asset : studio_assets()) {
        host.registerHandler(std::string(asset.path),
                             [asset](const HttpRequestPtr &, Reply &&reply) {
                                 reply(response(std::string(asset.body), 200, asset.mime));
                             },
                             {Get});
    }
    host.registerHandler("/favicon.ico",
                         [](const HttpRequestPtr &, Reply &&reply) { reply(response("", 204)); },
                         {Get});
    host.registerHandler("/api/v1/session",
                         [services](const HttpRequestPtr &, Reply &&reply) {
                             reply(response("{\"schema_version\":\"1.0\",\"session_token\":\"" +
                                            services->session + "\"}"));
                         },
                         {Get});
    host.registerHandler(
        "/api/v1/capabilities",
        [](const HttpRequestPtr &, Reply &&reply) { reply(response(application_capabilities())); },
        {Get});
    host.registerHandler(
        "/api/v1/models/status",
        [](const HttpRequestPtr &, Reply &&reply) { reply(response(model_validation_status())); },
        {Get});
    host.registerHandler("/api/v1/models/image", [services](const HttpRequestPtr &, Reply &&reply) {
        reply(response(services->jobs.model_status()));
    }, {Get});
    host.registerHandler("/api/v1/media", [services](const HttpRequestPtr &req, Reply &&reply) {
        services->dispatch(std::move(reply), [services, body=std::string(req->body()),
            name=utils::urlDecode(req->getHeader("x-file-name")), video=req->getHeader("x-media-kind")=="video"] { return services->jobs.import_media(body, name, video); });
    }, {Post});
    host.registerHandler("/api/v1/jobs", [services](const HttpRequestPtr &req, Reply &&reply) {
        if (req->method() == Post)
            services->dispatch(std::move(reply), [services, body=std::string(req->body())] { return services->jobs.submit(body); });
        else services->dispatch(std::move(reply), [services] { return services->jobs.list(); });
    }, {Get, Post});
    host.registerHandler("/api/v1/jobs/{1}", [services](const HttpRequestPtr &, Reply &&reply, std::string id) {
        services->dispatch(std::move(reply), [services, id] { return services->jobs.get(id); });
    }, {Get});
    host.registerHandler("/api/v1/jobs/{1}/cancel", [services](const HttpRequestPtr &, Reply &&reply, std::string id) {
        services->dispatch(std::move(reply), [services, id] { return services->jobs.cancel(id); });
    }, {Post});
    host.registerHandler("/api/v1/jobs/{1}/events", [services](const HttpRequestPtr &req, Reply &&reply, std::string id) {
        const auto after = req->getParameter("after");
        std::uint64_t sequence = 0;
        if (!after.empty()) {
            const auto parsed = std::from_chars(after.data(), after.data()+after.size(), sequence);
            if (parsed.ec != std::errc{} || parsed.ptr != after.data()+after.size() || sequence > 1000000) {
                reply(error(422, {"INVALID_CURSOR", "after", "Invalid event cursor"})); return;
            }
        }
        services->dispatch(std::move(reply), [services, id, sequence] { return services->jobs.events(id, sequence); });
    }, {Get});
    host.registerHandler("/api/v1/media/{1}", [services](const HttpRequestPtr &req, Reply &&reply, std::string id) {
        const auto file = services->jobs.media_file(id);
        if (is_error(file)) { reply(error(404, std::get<Error>(file))); return; }
        const auto path = std::get<std::filesystem::path>(file);
        std::ifstream input(path, std::ios::binary); unsigned char magic[2]{};
        input.read(reinterpret_cast<char *>(magic), 2);
        auto out = media_response(path,req,magic[0]==137?"image/png":magic[0]==255?"image/jpeg":magic[0]==0x1a?"video/webm":"video/mp4");
        reply(out);
    }, {Get});
    host.registerHandler("/api/v1/jobs/{1}/files/{2}", [services](const HttpRequestPtr &req, Reply &&reply, std::string id, std::string name) {
        const auto file = services->jobs.result_file(id, name);
        if (is_error(file)) { reply(error(404, std::get<Error>(file))); return; }
        auto out = media_response(std::get<std::filesystem::path>(file),req,name=="run.json"?"application/json":name.ends_with(".mp4")?"video/mp4":"image/png");
        if (req->getParameter("download") == "1") out->addHeader("Content-Disposition", "attachment; filename=\"seedvr2-"+id.substr(0,8)+"-"+name+"\"");
        reply(out);
    }, {Get});
    host.registerHandler("/api/v1/engine/devices",
                         [services](const HttpRequestPtr &, Reply &&reply) {
                             services->dispatch(std::move(reply), [] { return engine::devices(); });
                         }, {Get});
    host.registerHandler("/api/v1/engine/self-test",
                         [services](const HttpRequestPtr &req, Reply &&reply) {
                             services->dispatch(std::move(reply), [services, body = std::string(req->body())] {
                                 return services->application.self_test_and_save(body);
                             });
                         }, {Post});
    host.registerHandler(
        "/api/v1/models/policy",
        [](const HttpRequestPtr &, Reply &&reply) { reply(response(model_validation_policy())); },
        {Get});
    host.registerHandler("/api/v1/plan",
                         [services](const HttpRequestPtr &req, Reply &&reply) {
                             services->dispatch(std::move(reply),
                                                [body = std::string(req->body())] {
                                                    return evaluate_planning_request(body);
                                                });
                         },
                         {Post});
    host.registerHandler("/api/v1/plans",
                         [services](const HttpRequestPtr &req, Reply &&reply) {
                             services->dispatch(std::move(reply),
                                                [services, body = std::string(req->body())] {
                                                    return services->application.save_plan(body);
                                                });
                         },
                         {Post});
    host.registerHandler("/api/v1/records?kind={1}",
                         [services](const HttpRequestPtr &, Reply &&reply, std::string kind) {
                             services->dispatch(std::move(reply),
                                                [services, kind = std::move(kind)] {
                                                    return services->application.records(kind);
                                                });
                         },
                         {Get});
    host.registerHandler("/api/v1/records/{1}",
                         [services](const HttpRequestPtr &, Reply &&reply, std::string id) {
                             services->dispatch(std::move(reply), [services, id = std::move(id)] {
                                 return services->application.record(id);
                             });
                         },
                         {Get});
    host.setDefaultHandler([](const HttpRequestPtr &, Reply &&reply) {
        reply(error(404, {"HTTP_ERROR", "request", "No matching local endpoint"}));
    });
    host.registerBeginningAdvice([&host] {
        const auto addresses = host.getListeners();
        if (addresses.size() != 1)
            throw std::runtime_error("Unexpected local listeners");
        std::cout << "{\"event\":\"listening\",\"url\":\"http://" << addresses[0].toIpPort()
                  << "/\",\"inference\":true,\"framework\":\"Drogon\"}" << std::endl;
    });
    host.run();
    services->tasks.stop();
    return 0;
}
} // namespace seedvr2::server
