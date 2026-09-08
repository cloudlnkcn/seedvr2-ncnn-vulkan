#include "seedvr2/jobs.hpp"
#include "seedvr2/image.hpp"
#include "seedvr2/video.hpp"
#include "seedvr2/path.hpp"
#include "seedvr2/workspace.hpp"
#include "process.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <set>
#include <nlohmann/json.hpp>
#include <openssl/rand.h>
#include <sqlite3.h>
#include <stdexcept>
#include <thread>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace seedvr2 {
namespace {
using Json = nlohmann::json;
using Statement = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;
void sql(sqlite3 *db, const char *text) {
    if (sqlite3_exec(db, text, nullptr, nullptr, nullptr) != SQLITE_OK)
        throw std::runtime_error("Job database operation failed");
}
Statement query(sqlite3 *db, const char *text, const std::vector<std::string> &values = {}) {
    sqlite3_stmt *raw = nullptr;
    const int code = sqlite3_prepare_v2(db, text, -1, &raw, nullptr);
    Statement s(raw, sqlite3_finalize);
    if (code != SQLITE_OK) throw std::runtime_error("Job database query failed");
    for (std::size_t i = 0; i < values.size(); ++i)
        if (sqlite3_bind_text(s.get(), int(i+1), values[i].c_str(), int(values[i].size()), SQLITE_TRANSIENT) != SQLITE_OK)
            throw std::runtime_error("Job query binding failed");
    return s;
}
void done(const Statement &s) {
    if (sqlite3_step(s.get()) != SQLITE_DONE) throw std::runtime_error("Job write failed");
}
std::string column(const Statement &s, int i = 0) {
    const auto *text = sqlite3_column_text(s.get(), i);
    return text ? reinterpret_cast<const char *>(text) : "";
}
std::string identity() {
    unsigned char bytes[16];
    if (RAND_bytes(bytes, 16) != 1) throw std::runtime_error("Cannot allocate job identity");
    constexpr char hex[] = "0123456789abcdef";
    std::string value;
    for (auto b : bytes) { value += hex[b>>4]; value += hex[b&15]; }
    return value;
}
void valid_id(std::string_view id) {
    if (id.size() != 32 || id.find_first_not_of("0123456789abcdef") != std::string_view::npos)
        throw std::runtime_error("Invalid workspace identity");
}
std::string now() {
    const auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    char text[32]; std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text;
}
bool terminal(std::string_view status) {
    return status == "SUCCEEDED" || status == "FAILED" || status == "CANCELLED" || status == "INTERRUPTED";
}
void write_json(const std::filesystem::path &path, const Json &doc) {
    std::ofstream out(path); out << doc.dump() << '\n'; out.close();
    if (!out) throw std::runtime_error("Cannot save worker request");
}
Json strict_request(std::string_view body) {
    if (body.size() > 4096) throw std::runtime_error("Job request exceeds 4 KiB");
    unsigned keys = 0;
    auto doc = Json::parse(body, [&](int depth, Json::parse_event_t event, Json &) {
        if (depth > 1) throw std::runtime_error("Job request must be a flat object");
        if (event == Json::parse_event_t::key) ++keys;
        return true;
    });
    const bool video=doc.value("schema_version","")=="seedvr2-video-job-v1";
    std::set<std::string> expected{"schema_version", "media_id", "size", "backend", "gpu", "seed"};
    if (video) expected.insert("max_frames");
    if (!doc.is_object() || doc.size() != expected.size() || keys != expected.size())
        throw std::runtime_error("Unexpected or duplicate job fields");
    for (const auto &[key, value] : doc.items()) {
        (void)value;
        if (!expected.contains(key)) throw std::runtime_error("Unknown job field");
    }
    valid_id(doc.at("media_id").get<std::string>());
    if ((!video && doc.at("schema_version") != "seedvr2-image-job-v1") ||
        (doc.at("backend") != "cpu" && doc.at("backend") != "vulkan") ||
        !doc.at("size").is_number_integer() || doc.at("size") < 64 || doc.at("size") > 512 || doc.at("size").get<int>()%16 ||
        !doc.at("gpu").is_number_integer() || doc.at("gpu") < -1 || doc.at("gpu") > 64 ||
        !doc.at("seed").is_number_integer() || doc.at("seed") < 0 || doc.at("seed") > 4294967295ULL)
        throw std::runtime_error("Invalid image size, backend, device or seed");
    if (video && (doc.at("size")>128 || !doc.at("max_frames").is_number_integer() || doc.at("max_frames")<1 || doc.at("max_frames")>17))
        throw std::runtime_error("Video preview accepts at most 17 frames and 128 pixels long side");
    return doc;
}
}
struct Jobs::Impl {
    sqlite3 *db = nullptr;
    std::filesystem::path root, worker, model, video_model;
    std::mutex mutex;
    std::condition_variable wake;
    std::atomic<bool> stop{false}, cancel_active{false};
    std::string active;
    std::thread supervisor;
#if defined(__linux__)
    int lock_file = -1;
#endif
    ~Impl() {
        stop = true; wake.notify_all();
        if (supervisor.joinable()) supervisor.join();
        if (db) sqlite3_close(db);
#if defined(__linux__)
        if (lock_file >= 0) close(lock_file);
#endif
    }
    Json read_job(const std::string &id) {
        auto s = query(db, "SELECT payload FROM image_jobs WHERE id=?", {id});
        if (sqlite3_step(s.get()) != SQLITE_ROW) throw std::runtime_error("Job does not exist");
        return Json::parse(column(s));
    }
    Json read_media(const std::string &id) {
        auto s = query(db, "SELECT payload FROM image_media WHERE id=?", {id});
        if (sqlite3_step(s.get()) != SQLITE_ROW) throw std::runtime_error("Imported image does not exist");
        return Json::parse(column(s));
    }
    void update(Json &job, std::string_view type, const Json &details) {
        const auto id = job.at("id").get<std::string>();
        const auto seq = job.at("sequence").get<std::uint64_t>()+1;
        job["sequence"] = seq; job["updated_at"] = now();
        Json event{{"sequence", seq}, {"type", type}, {"at", job["updated_at"]}, {"details", details}, {"status", job["status"]}};
        sql(db, "BEGIN IMMEDIATE");
        try {
            auto s = query(db, "INSERT INTO image_job_events(job_id,sequence,payload) VALUES(?,?,?)", {id, std::to_string(seq), event.dump()});
            done(s);
            s = query(db, "UPDATE image_jobs SET status=?,payload=? WHERE id=?", {job.at("status").get<std::string>(), job.dump(), id});
            done(s); sql(db, "COMMIT");
        } catch (...) { sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr); throw; }
    }
    void loop() {
        for (;;) {
            Json job;
            {
                std::unique_lock lock(mutex);
                wake.wait_for(lock, std::chrono::milliseconds(250));
                if (stop) return;
                auto s = query(db, "SELECT payload FROM image_jobs WHERE status='QUEUED' ORDER BY rowid LIMIT 1");
                if (sqlite3_step(s.get()) != SQLITE_ROW) continue;
                job = Json::parse(column(s)); s.reset();
                active = job.at("id").get<std::string>(); cancel_active = false;
                job["status"] = "RUNNING"; job["progress"] = {{"stage", "starting"}, {"completed", 0}, {"total", 38}};
                update(job, "started", job["progress"]);
            }
            Json result, failure;
            int exit_code = -1;
            try {
                const auto id = job.at("id").get<std::string>();
                const auto directory = root/"jobs"/id;
                std::filesystem::create_directories(directory);
                auto request = job.at("request");
                const bool video=request.at("schema_version")=="seedvr2-video-job-v1";
                request["kind"]=video?"video":"image";
                request["schema_version"] = "seedvr2-worker-request-v1";
                request["model"] = (video?video_model:model).string(); request["input"] = (root/"media"/request.at("media_id").get<std::string>()/"input").string();
                request["output"] = (directory/"result").string(); request["threads"] = 4;
                write_json(directory/"request.json", request);
                std::vector<std::string> arguments{worker.string(), "--request", (directory/"request.json").string()};
#if defined(__linux__)
                arguments.insert(arguments.end(), {"--parent-pid", std::to_string(getpid())});
#endif
                exit_code = jobs_detail::run_process(arguments, directory/"worker.log", [&] { return stop || cancel_active; },
                    [&](const std::string &line) {
                        const auto event = Json::parse(line);
                        if (event.at("protocol") != "seedvr2-worker-v1") throw std::runtime_error("Worker protocol mismatch");
                        const auto type = event.at("type").get<std::string>();
                        if (type == "result") {
                            if (!result.is_null() || !failure.is_null()) throw std::runtime_error("Duplicate terminal worker message");
                            result = event.at("report");
                        } else if (type == "error") failure = event;
                        else if (type == "progress") {
                            if (!result.is_null() || !failure.is_null() || event.at("total") != 38 ||
                                !event.at("completed").is_number_integer() || event.at("completed") < 0 || event.at("completed") > 38)
                                throw std::runtime_error("Invalid worker progress");
                            std::lock_guard lock(mutex);
                            auto fresh = read_job(id);
                            if (event.at("completed") < fresh.at("progress").at("completed")) throw std::runtime_error("Worker progress moved backwards");
                            fresh["progress"] = event;
                            update(fresh, "progress", event);
                        } else throw std::runtime_error("Unknown worker event");
                    });
                if (exit_code == 0 && !result.is_null()) {
                    if (result.at("status") != "SUCCEEDED" || result.at("stages").size() != 36 || result.at("output").at("path") != (video?"output.mp4":"output.png") ||
                        result.at("schema_version")!=(video?"seedvr2-video-run-v1":"seedvr2-image-run-v1"))
                        throw std::runtime_error("Worker did not return a complete image run");
                    const auto inspected = video?engine::inspect_video(directory/"result/output.mp4"):engine::inspect_image(directory/"result/output.png");
                    if (is_error(inspected) || Json::parse(std::get<std::string>(inspected)).at("sha256") != result.at("output").at("sha256"))
                        throw std::runtime_error("Worker output identity mismatch");
                }
            } catch (const std::exception &e) { failure = {{"code", "WORKER_FAILED"}, {"message", e.what()}}; }
            {
                std::lock_guard lock(mutex);
                auto final = read_job(active);
                if (stop) final["status"] = "INTERRUPTED";
                else if (cancel_active || final.at("status") == "CANCELLING") final["status"] = "CANCELLED";
                else if (exit_code == 0 && !result.is_null() && failure.is_null()) { final["status"] = "SUCCEEDED"; final["result"] = result; }
                else { final["status"] = "FAILED"; final["error"] = failure.is_null() ? Json{{"code", "WORKER_EXIT"}, {"message", "Worker exited without a complete result"}} : failure; }
                final["exit_code"] = exit_code;
                update(final, "finished", {{"exit_code", exit_code}, {"error", final.value("error", Json(nullptr))}});
                active.clear(); cancel_active = false;
            }
        }
    }
};
Jobs::Jobs(const std::filesystem::path &database, const std::filesystem::path &worker, const std::filesystem::path &model, const std::filesystem::path &video_model)
    : impl_(std::make_unique<Impl>()) {
    Workspace migrate(database);
    auto &p = *impl_;
    p.root = std::filesystem::absolute(database).parent_path()/(database.stem().string()+"-files");
    p.worker = std::filesystem::absolute(worker); p.model = std::filesystem::absolute(model);
    p.video_model=video_model.empty()?std::filesystem::path{}:std::filesystem::absolute(video_model);
    std::filesystem::create_directories(p.root);
    std::filesystem::permissions(p.root, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);
#if defined(__linux__)
    p.lock_file = open((p.root/"supervisor.lock").c_str(), O_RDWR|O_CREAT|O_CLOEXEC, 0600);
    if (p.lock_file < 0 || flock(p.lock_file, LOCK_EX|LOCK_NB) != 0)
        throw std::runtime_error("This workspace already has an active job supervisor");
#endif
    if (sqlite3_open_v2(database.string().c_str(), &p.db, SQLITE_OPEN_READWRITE|SQLITE_OPEN_FULLMUTEX, nullptr) != SQLITE_OK)
        throw std::runtime_error("Cannot open job workspace");
    sqlite3_busy_timeout(p.db, 5000);
    sql(p.db, "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;");
    sql(p.db, "BEGIN IMMEDIATE; CREATE TABLE IF NOT EXISTS image_media(id TEXT PRIMARY KEY,payload TEXT NOT NULL);"
              "CREATE TABLE IF NOT EXISTS image_jobs(id TEXT PRIMARY KEY,status TEXT NOT NULL CHECK(status IN ('QUEUED','RUNNING','CANCELLING','SUCCEEDED','FAILED','CANCELLED','INTERRUPTED')),payload TEXT NOT NULL);"
              "CREATE INDEX IF NOT EXISTS image_jobs_status ON image_jobs(status);"
              "CREATE TABLE IF NOT EXISTS image_job_events(job_id TEXT NOT NULL REFERENCES image_jobs(id),sequence INTEGER NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(job_id,sequence)); PRAGMA user_version=3; COMMIT;");
    std::vector<Json> recovery;
    auto s = query(p.db, "SELECT payload FROM image_jobs WHERE status IN ('QUEUED','RUNNING','CANCELLING')");
    while (sqlite3_step(s.get()) == SQLITE_ROW) recovery.push_back(Json::parse(column(s)));
    s.reset();
    for (auto &job : recovery) {
        job["status"] = "INTERRUPTED";
        job["error"] = {{"code", "SERVICE_RESTARTED"}, {"message", "Processing was interrupted; retry creates a fresh run"}};
        p.update(job, "recovered", job["error"]);
    }
    p.supervisor = std::thread([&p] {
        try { p.loop(); }
        catch (const std::exception &) { p.stop = true; }
    });
}
Jobs::~Jobs() = default;
Result<std::string> Jobs::import_media(std::string_view bytes, std::string_view name, bool video) {
    try {
        if (bytes.empty() || bytes.size() > (video?256ULL:32ULL)*1024*1024 || name.empty() || name.size() > 512 || name.find_first_of("\r\n") != std::string_view::npos)
            throw std::runtime_error("Choose an image up to 32 MiB or a video up to 256 MiB");
        const auto id = identity(); const auto folder = impl_->root/"media"/id;
        std::filesystem::create_directories(folder);
        try {
            std::ofstream out(folder/"input", std::ios::binary);
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size())); out.close();
            if (!out) throw std::runtime_error("Cannot save imported image");
            auto inspected = video?engine::inspect_video(folder/"input"):engine::inspect_image(folder/"input");
            if (is_error(inspected)) throw std::runtime_error(std::get<Error>(inspected).message);
            auto media = Json::parse(std::get<std::string>(inspected));
            media["id"] = id; media["name"] = name; media["created_at"] = now();
            media["kind"]=video?"video":"image";
            media["url"] = "/api/v1/media/"+id;
            std::lock_guard lock(impl_->mutex);
            auto s = query(impl_->db, "INSERT INTO image_media(id,payload) VALUES(?,?)", {id, media.dump()}); done(s);
            return media.dump();
        } catch (...) { std::filesystem::remove_all(folder); throw; }
    } catch (const std::exception &e) { return Error{"IMAGE_IMPORT_FAILED", "image", e.what()}; }
}
Result<std::string> Jobs::submit(std::string_view body) {
    try {
        auto request = strict_request(body);
        const bool video=request.at("schema_version")=="seedvr2-video-job-v1";
        const auto model=video?impl_->video_model:impl_->model;
        if (model.empty() || !std::filesystem::is_regular_file(impl_->worker) || !std::filesystem::is_regular_file(model/"manifest.json"))
            throw std::runtime_error("Install the complete image model package before starting");
        std::lock_guard lock(impl_->mutex);
        if (impl_->stop) throw std::runtime_error("Job supervisor is unavailable; restart the local service");
        auto media = impl_->read_media(request.at("media_id").get<std::string>());
        if ((media.value("kind","image")=="video")!=video) throw std::runtime_error("Job type does not match the imported media");
        const double ratio = double(std::min(media.at("width").get<int>(), media.at("height").get<int>()))/
            std::max(media.at("width").get<int>(), media.at("height").get<int>());
        if (int(std::nearbyint(request.at("size").get<int>()*ratio))/16*16 < 64)
            throw std::runtime_error("The output short side would be below 64 pixels; choose a larger size");
        auto count = query(impl_->db, "SELECT count(*) FROM image_jobs WHERE status IN ('QUEUED','RUNNING','CANCELLING')");
        if (sqlite3_step(count.get()) != SQLITE_ROW || sqlite3_column_int(count.get(), 0) >= 8)
            throw std::runtime_error("The processing queue is full (8 jobs)");
        count.reset();
        const auto id = identity(), at = now();
        Json job{{"schema_version", request.at("schema_version")}, {"id", id}, {"status", "QUEUED"},
            {"request", request}, {"input", media}, {"created_at", at}, {"updated_at", at},
            {"sequence", 0}, {"progress", {{"stage", "queued"}, {"completed", 0}, {"total", 38}}},
            {"result", nullptr}, {"error", nullptr}};
        auto s = query(impl_->db, "INSERT INTO image_jobs(id,status,payload) VALUES(?,'QUEUED',?)", {id, job.dump()}); done(s);
        impl_->update(job, "queued", request);
        impl_->wake.notify_one();
        return job.dump();
    } catch (const std::exception &e) { return Error{"JOB_INVALID", "job", e.what()}; }
}
Result<std::string> Jobs::list() {
    try {
        std::lock_guard lock(impl_->mutex); Json items = Json::array();
        auto s = query(impl_->db, "SELECT payload FROM image_jobs ORDER BY rowid DESC LIMIT 100");
        while (sqlite3_step(s.get()) == SQLITE_ROW) {
            auto job = Json::parse(column(s));
            if (!job["result"].is_null()) job["result"].erase("stages");
            items.push_back(job);
        }
        return Json{{"items", items}, {"limit", 100}}.dump();
    } catch (const std::exception &e) { return Error{"JOB_READ_FAILED", "jobs", e.what()}; }
}
Result<std::string> Jobs::get(std::string_view id) {
    try { valid_id(id); std::lock_guard lock(impl_->mutex); return impl_->read_job(std::string(id)).dump(); }
    catch (const std::exception &e) { return Error{"JOB_NOT_FOUND", "id", e.what()}; }
}
Result<std::string> Jobs::cancel(std::string_view id) {
    try {
        valid_id(id); std::lock_guard lock(impl_->mutex);
        auto job = impl_->read_job(std::string(id));
        const auto status = job.at("status").get<std::string>();
        if (!terminal(status) && status != "CANCELLING") {
            job["status"] = status == "QUEUED" ? "CANCELLED" : "CANCELLING";
            if (impl_->active == id) impl_->cancel_active = true;
            impl_->update(job, "cancel_requested", Json::object());
        }
        return job.dump();
    } catch (const std::exception &e) { return Error{"JOB_NOT_FOUND", "id", e.what()}; }
}
Result<std::string> Jobs::events(std::string_view id, std::uint64_t after) {
    try {
        valid_id(id); std::lock_guard lock(impl_->mutex);
        const auto job = impl_->read_job(std::string(id));
        auto s = query(impl_->db, "SELECT payload FROM image_job_events WHERE job_id=? AND sequence>? ORDER BY sequence LIMIT 256", {std::string(id), std::to_string(after)});
        Json items = Json::array();
        while (sqlite3_step(s.get()) == SQLITE_ROW) items.push_back(Json::parse(column(s)));
        return Json{{"items", items}, {"sequence", job["sequence"]}, {"status", job["status"]}}.dump();
    } catch (const std::exception &e) { return Error{"JOB_NOT_FOUND", "id", e.what()}; }
}
Result<std::filesystem::path> Jobs::media_file(std::string_view id) {
    try { valid_id(id); std::lock_guard lock(impl_->mutex); impl_->read_media(std::string(id)); return impl_->root/"media"/std::string(id)/"input"; }
    catch (const std::exception &e) { return Error{"MEDIA_NOT_FOUND", "id", e.what()}; }
}
Result<std::filesystem::path> Jobs::result_file(std::string_view id, std::string_view name) {
    try {
        valid_id(id);
        if (name != "output.png" && name != "comparison-input.png" && name != "run.json" && name != "output.mp4" && name != "comparison-input.mp4") throw std::runtime_error("Unknown result file");
        std::lock_guard lock(impl_->mutex);
        if (impl_->read_job(std::string(id)).at("status") != "SUCCEEDED") throw std::runtime_error("This job has no completed result");
        const auto path=impl_->root/"jobs"/std::string(id)/"result"/std::string(name);
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("This result file is unavailable");
        return path;
    } catch (const std::exception &e) { return Error{"RESULT_NOT_FOUND", "result", e.what()}; }
}
std::string Jobs::model_status() {
    return Json{{"installed", std::filesystem::is_regular_file(impl_->model/"manifest.json")},
        {"worker_available", std::filesystem::is_regular_file(impl_->worker) && !impl_->stop},
        {"profile", "seedvr2-3b-image-fp32-b-v1"}, {"model_verified", false},
        {"integrity", "CHECKED_BY_WORKER_BEFORE_EVERY_RUN"}, {"sizes", {128, 256, 384, 512}},
        {"image", true}, {"video", true},
        {"video_model",{{"installed",!impl_->video_model.empty() && std::filesystem::is_regular_file(impl_->video_model/"manifest.json")},
            {"profile","seedvr2-3b-video-fp32-b-v1"},{"sizes",{64,96,128}},{"max_frames",17},{"streaming_cache",false},{"audio",false}}},
        {"max_queued", 8}, {"concurrency", 1}}.dump();
}
} // namespace seedvr2
