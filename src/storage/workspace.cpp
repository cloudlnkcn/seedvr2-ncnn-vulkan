#include "seedvr2/workspace.hpp"
#include "seedvr2/path.hpp"
#include <cstdlib>
#include <mutex>
#include <nlohmann/json.hpp>
#include <openssl/rand.h>
#include <sqlite3.h>
#include <stdexcept>

namespace seedvr2 {
namespace {
using Json = nlohmann::json;
using Statement = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;
void exec(sqlite3 *db, const char *sql) {
    if (sqlite3_exec(db, sql, nullptr, nullptr, nullptr) != SQLITE_OK)
        throw std::runtime_error("Workspace database operation failed");
}
Statement statement(sqlite3 *db, const char *sql) {
    sqlite3_stmt *raw = nullptr;
    const int code = sqlite3_prepare_v2(db, sql, -1, &raw, nullptr);
    Statement result(raw, sqlite3_finalize);
    if (code != SQLITE_OK)
        throw std::runtime_error("Workspace query failed");
    return result;
}
void bind_text(sqlite3_stmt *s, int index, std::string_view text) {
    if (sqlite3_bind_text(s, index, text.data(), static_cast<int>(text.size()), SQLITE_TRANSIENT) !=
        SQLITE_OK)
        throw std::runtime_error("Workspace parameter failed");
}
std::string column(sqlite3_stmt *s, int i) {
    const auto *value = sqlite3_column_text(s, i);
    return value ? reinterpret_cast<const char *>(value) : "";
}
std::string new_id() {
    unsigned char bytes[16];
    if (RAND_bytes(bytes, 16) != 1)
        throw std::runtime_error("Cannot allocate record identity");
    constexpr char hex[] = "0123456789abcdef";
    std::string id;
    for (auto b : bytes) {
        id += hex[b >> 4];
        id += hex[b & 15];
    }
    return id;
}
bool valid_kind(std::string_view kind) {
    return kind == "plan" || kind == "model-audit" || kind == "operator-test";
}
} // namespace
struct Workspace::Impl {
    sqlite3 *database = nullptr;
    std::mutex mutex;
    ~Impl() {
        if (database)
            sqlite3_close(database);
    }
};
Workspace::Workspace(const std::filesystem::path &database) : impl_(std::make_unique<Impl>()) {
    if (database.has_parent_path())
        std::filesystem::create_directories(database.parent_path());
    const auto utf8 = database.u8string();
    if (sqlite3_open_v2(reinterpret_cast<const char *>(utf8.c_str()), &impl_->database,
                        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
                        nullptr) != SQLITE_OK)
        throw std::runtime_error("Cannot open workspace database");
    sqlite3_busy_timeout(impl_->database, 5000);
    exec(impl_->database,
         "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;");
    exec(impl_->database, "BEGIN IMMEDIATE");
    try {
        auto version = statement(impl_->database, "PRAGMA user_version");
        if (sqlite3_step(version.get()) != SQLITE_ROW)
            throw std::runtime_error("Cannot read workspace version");
        const int current = sqlite3_column_int(version.get(), 0);
        version.reset();
        if (current > 3)
            throw std::runtime_error("Workspace uses a newer schema; downgrade is refused");
        if (current == 0) {
            exec(impl_->database, "CREATE TABLE records(id TEXT PRIMARY KEY, kind TEXT NOT NULL "
                                  "CHECK(kind IN ('plan','model-audit')), created_at TEXT NOT NULL "
                                  "DEFAULT(strftime('%Y-%m-%dT%H:%M:%fZ','now')), status TEXT NOT "
                                  "NULL, payload TEXT NOT NULL); CREATE INDEX records_kind_created "
                                  "ON records(kind,created_at DESC); PRAGMA user_version=1;");
        }
        if (current <= 1) {
            exec(impl_->database,
                 "CREATE TABLE records_v2(id TEXT PRIMARY KEY, kind TEXT NOT NULL "
                 "CHECK(kind IN ('plan','model-audit','operator-test')), created_at TEXT NOT NULL "
                 "DEFAULT(strftime('%Y-%m-%dT%H:%M:%fZ','now')), status TEXT NOT NULL, payload TEXT NOT NULL);"
                 "INSERT INTO records_v2 SELECT id,kind,created_at,status,payload FROM records ORDER BY rowid;"
                 "DROP TABLE records; ALTER TABLE records_v2 RENAME TO records;"
                 "CREATE INDEX records_kind_created ON records(kind,created_at DESC); PRAGMA user_version=2;");
        }
        exec(impl_->database, "COMMIT");
    } catch (...) {
        sqlite3_exec(impl_->database, "ROLLBACK", nullptr, nullptr, nullptr);
        throw;
    }
}
Workspace::~Workspace() = default;
Result<std::string> Workspace::save(std::string_view kind, std::string_view status,
                                    std::string_view payload) {
    if (!valid_kind(kind) || payload.size() > 8 * 1024 * 1024)
        return Error{"RECORD_INVALID", "record", "Unsupported record kind or record exceeds 8 MiB"};
    std::lock_guard lock(impl_->mutex);
    const auto id = new_id();
    auto s =
        statement(impl_->database, "INSERT INTO records(id,kind,status,payload) VALUES(?,?,?,?)");
    bind_text(s.get(), 1, id);
    bind_text(s.get(), 2, kind);
    bind_text(s.get(), 3, status);
    bind_text(s.get(), 4, payload);
    if (sqlite3_step(s.get()) != SQLITE_DONE)
        return Error{"WORKSPACE_WRITE", "record", "Record could not be saved"};
    return Json{{"schema_version", "1.0"}, {"id", id}, {"kind", kind}, {"status", status}}.dump();
}
Result<std::string> Workspace::list(std::string_view kind) {
    if (!valid_kind(kind))
        return Error{"RECORD_INVALID", "kind", "Unknown record kind"};
    std::lock_guard lock(impl_->mutex);
    auto s = statement(impl_->database, "SELECT id,kind,created_at,status FROM records WHERE "
                                        "kind=? ORDER BY created_at DESC,rowid DESC LIMIT 100");
    bind_text(s.get(), 1, kind);
    Json rows = Json::array();
    int code = 0;
    while ((code = sqlite3_step(s.get())) == SQLITE_ROW)
        rows.push_back({{"id", column(s.get(), 0)},
                        {"kind", column(s.get(), 1)},
                        {"created_at", column(s.get(), 2)},
                        {"status", column(s.get(), 3)}});
    if (code != SQLITE_DONE)
        return Error{"WORKSPACE_READ", "records", "Records could not be read"};
    return Json{{"schema_version", "1.0"}, {"items", rows}, {"limit", 100}}.dump();
}
Result<std::string> Workspace::get(std::string_view id) {
    if (id.size() != 32 || id.find_first_not_of("0123456789abcdef") != std::string_view::npos)
        return Error{"RECORD_INVALID", "id", "Expected a 32-character record id"};
    std::lock_guard lock(impl_->mutex);
    auto s = statement(impl_->database, "SELECT payload FROM records WHERE id=?");
    bind_text(s.get(), 1, id);
    const int code = sqlite3_step(s.get());
    if (code == SQLITE_DONE)
        return Error{"RECORD_NOT_FOUND", "id", "Record does not exist"};
    if (code != SQLITE_ROW)
        return Error{"WORKSPACE_READ", "record", "Record could not be read"};
    return column(s.get(), 0);
}
std::filesystem::path default_database_path() {
#if defined(_WIN32)
    const char *base = std::getenv("LOCALAPPDATA");
    if (base && *base)
        return seedvr2::utf8_path(base) / "seedvr2/workspace.sqlite3";
#else
    const char *data = std::getenv("XDG_DATA_HOME");
    if (data && *data && std::filesystem::path(data).is_absolute())
        return std::filesystem::path(data) / "seedvr2/workspace.sqlite3";
    const char *base = std::getenv("HOME");
    if (base && *base)
        return std::filesystem::path(base) / ".local/share/seedvr2/workspace.sqlite3";
#endif
    throw std::runtime_error("No local data directory found; specify --database");
}
} // namespace seedvr2
