#!/usr/bin/env python3
"""Real HTTP boundary and CLI parity checks. Development tool, no third-party packages."""
import argparse
import concurrent.futures
import json
import re
import tempfile
from pathlib import Path
import queue
import subprocess
import sqlite3
import threading
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server", type=Path)
    parser.add_argument("cli", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, help="Also run the real Vulkan self-test on this device")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    request_text = (root / "examples/plan-720p.json").read_text()
    request = json.loads(request_text)
    temporary = tempfile.TemporaryDirectory(prefix="seedvr2-web-check-")
    database = Path(temporary.name) / "workspace.sqlite3"
    migrated_id = 'a' * 32
    migrated_payload = '{"document_type":"migration-fixture","value":"中文旧记录"}'
    with sqlite3.connect(database) as connection:
        connection.executescript("CREATE TABLE records(id TEXT PRIMARY KEY, kind TEXT NOT NULL "
            "CHECK(kind IN ('plan','model-audit')), created_at TEXT NOT NULL "
            "DEFAULT(strftime('%Y-%m-%dT%H:%M:%fZ','now')), status TEXT NOT NULL, payload TEXT NOT NULL);"
            "CREATE INDEX records_kind_created ON records(kind,created_at DESC); PRAGMA user_version=1;")
        connection.execute("INSERT INTO records(id,kind,created_at,status,payload) VALUES(?,?,?,?,?)",
                           (migrated_id, 'plan', '2026-01-01T00:00:00.000Z', 'PLANNED', migrated_payload))
    model_dir = Path(temporary.name)/"声明模型"
    model_dir.mkdir()
    manifest=model_dir/"manifest.json"
    manifest.write_text(json.dumps({"profile":"seedvr2-3b-image-dit-fp16-storage-v1",
        "storage_precision":{"dit_linear_weights":"fp16-ieee","activation":"fp32","arithmetic":"fp32"}}))
    process = subprocess.Popen([str(args.server.resolve()), "--port", "0", "--database", str(database), "--model", str(model_dir)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    results = []
    try:
        startup = queue.Queue(maxsize=1)
        threading.Thread(target=lambda: startup.put(process.stdout.readline()), daemon=True).start()
        url = json.loads(startup.get(timeout=15))["url"].rstrip("/")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def call(path, body=None, headers=None, method=None):
            data = body.encode() if isinstance(body, str) else body
            http = urllib.request.Request(url + path, data=data, headers=headers or {}, method=method)
            try:
                response = opener.open(http, timeout=15)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                return response.status, response.read(), response.headers

        def check(name, condition):
            results.append({"case": name, "status": "PASS" if condition else "FAIL"})
            if not condition:
                raise AssertionError(name)

        status, body, headers = call("/")
        check("embedded-offline-page", status == 200 and "本地工作台" in body.decode())
        check("local-response-policy", headers.get("Cache-Control") == "no-store" and
              "frame-ancestors 'none'" in headers.get("Content-Security-Policy", "") and
              headers.get("Access-Control-Allow-Origin") is None)
        check("external-link-can-open-home", call("/", headers={"Sec-Fetch-Site": "cross-site",
              "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"})[0] == 200)
        asset_paths = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', body.decode())
        check("vite-assets-embedded", len(asset_paths) >= 2)
        for path in asset_paths:
            check("asset-" + path[1:], call(path)[0] == 200)
        status, body, _ = call("/api/v1/capabilities")
        caps = json.loads(body)
        check("truthful-capabilities", status == 200 and caps["planning"] and caps["inference"]
              and caps["gpu_probe"] and caps['ncnn_linked'] and caps['operator_self_test']
              and caps['engine']['vae_image_diagnostics'] and caps['engine']['dit_block_diagnostics']
              and caps["persistent_queue"] and caps["events"] == "DURABLE_CURSOR_POLLING" and not caps['model_certification'])
        token = json.loads(call("/api/v1/session")[1])["session_token"]
        check("session-token", len(token) == 64 and all(c in "0123456789abcdef" for c in token))
        auth = {"Content-Type": "application/json", "X-SeedVR2-Session": token}
        check('v1-record-retained', call('/api/v1/records/'+migrated_id)[1].decode() == migrated_payload)
        with sqlite3.connect(database) as connection:
            check('transactional-schema-v3', connection.execute('PRAGMA user_version').fetchone()[0] == 3
                  and connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok')
        status, body, _ = call("/api/v1/plan", request_text, auth)
        expected = json.loads(subprocess.check_output([str(args.cli.resolve()), "plan", "--request",
                                                       str(root / "examples/plan-720p.json")], text=True))
        check("http-cli-full-response-parity", status == 200 and json.loads(body) == expected)
        image_request = {**request, "media": {"kind": "image", "width": 641, "height": 361, "frames": 1}}
        status, body, _ = call("/api/v1/plan", json.dumps(image_request), auth)
        planned = json.loads(body)
        check("non-aligned-image-geometry", status == 200 and
              planned["logical_output"] == {"width": 1282, "height": 722, "frames": 1} and
              planned["working_extent"] == {"width": 1296, "height": 736, "frames": 1} and
              not planned["runnable"] and planned["memory"]["total_peak_bytes"] is None)
        cases = [
            ("missing-session", request_text, {"Content-Type": "application/json"}, 403, "SESSION_REQUIRED"),
            ("wrong-session", request_text, {**auth, "X-SeedVR2-Session": "bad"}, 403, "SESSION_REQUIRED"),
            ("foreign-origin", request_text, {**auth, "Origin": "https://example.org"}, 403, "ORIGIN_REJECTED"),
            ("null-origin", request_text, {**auth, "Origin": "null"}, 403, "ORIGIN_REJECTED"),
            ("dns-rebinding-host", request_text, {**auth, "Host": "example.org"}, 403, "HOST_REJECTED"),
            ("cross-site-metadata", request_text, {**auth, "Sec-Fetch-Site": "cross-site"}, 403, "ORIGIN_REJECTED"),
            ("wrong-content-type", request_text, {**auth, "Content-Type": "text/plain"}, 415, "CONTENT_TYPE"),
            ("malformed-json", "{", auth, 422, "INVALID_JSON"),
            ("unsupported-schema", '{"schema_version":"0.2-design"}', auth, 422, "SCHEMA_UNSUPPORTED"),
            ("duplicate-json-key", '{"schema_version":"1.0","schema_version":"1.0"}', auth, 422, "DUPLICATE_KEY"),
            ("bounded-request-body", " " * (1024 * 1024 + 1), auth, 413, "REQUEST_TOO_LARGE"),
        ]
        for name, data, case_headers, expected_status, code in cases:
            status, body, _ = call("/api/v1/plan", data, case_headers)
            check(name, status == expected_status and ((status == 413 and not body) or json.loads(body)["error"]["code"] == code))
        check("same-origin-browser-post", call("/api/v1/plan", request_text,
              {**auth, "Origin": url, "Sec-Fetch-Site": "same-origin"})[0] == 200)
        check("foreign-session-read-rejected", call("/api/v1/session", headers={"Origin": "https://example.org"})[0] == 403)
        check("foreign-navigation-cannot-read-session", call("/api/v1/session", headers={
              "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"})[0] == 403)
        check("no-filesystem-mount", call("/etc/passwd")[0] == 404)
        check("reject-incomplete-image-job", call("/api/v1/jobs", "{}", auth)[0] == 422)
        status, body, _ = call("/api/v1/models/status")
        model = json.loads(body)
        expected_model = json.loads(subprocess.check_output([str(args.cli.resolve()), "models", "status"], text=True))
        check("web-cli-model-status-parity", status == 200 and model == expected_model)
        declared=json.loads(call("/api/v1/models/image")[1])
        check("storage-model-profile-declared",declared["profile"]=="seedvr2-3b-image-dit-fp16-storage-v1" and declared["storage_precision"]["dit_linear_weights"]=="fp16-ieee")
        check("declaration-is-not-certification",declared["model_verified"] is False and declared["metadata_status"]=="DECLARED_UNVERIFIED")
        manifest.write_text("{")
        invalid=json.loads(call("/api/v1/models/image")[1])
        check("invalid-model-metadata-is-not-ready",invalid["installed"] is False and invalid["metadata_status"]=="INVALID")
        manifest.write_text(" "*(256*1024+1))
        check("model-metadata-size-bounded",json.loads(call("/api/v1/models/image")[1])["metadata_status"]=="INVALID")

        check("model-cannot-be-certified", not model["model_verified"] and model["certificate"] is None and len(model["gates"]) == 12)
        check("policy-needs-calibration", json.loads(call("/api/v1/models/policy")[1])["calibration_status"] == "NOT_FROZEN")
        status, body, _ = call("/api/v1/plans", request_text, auth)
        record = json.loads(body)
        check("save-plan-is-not-inference", status == 200 and record["status"] == "PLANNED")
        status, body, _ = call("/api/v1/records?kind=plan")
        check("list-persisted-plan", status == 200 and json.loads(body)["items"][0]["id"] == record["id"])
        detail = json.loads(call("/api/v1/records/" + record["id"])[1])
        check("saved-plan-preserves-shared-result", detail["plan"] == expected and detail["execution_status"] == "NOT_RUN")
        from_cli = json.loads(subprocess.check_output([str(args.cli.resolve()), "--database", str(database), "history", "get", "--id", record["id"]], text=True))
        check("cli-can-read-web-record-without-server-api", from_cli == detail)
        audit = subprocess.run([str(args.cli.resolve()), "--database", str(database), "models", "audit", "--bundle", str(root / "examples/model-evidence-empty"), "--save"], capture_output=True, text=True)
        audit_record = json.loads(audit.stdout)
        check("cli-audit-is-visible-in-web", audit.returncode == 6 and json.loads(call("/api/v1/records?kind=model-audit")[1])["items"][0]["id"] == audit_record["record_id"])
        web_audit = json.loads(call("/api/v1/records/" + audit_record["record_id"])[1])
        check("web-audit-retains-blocked-verdict", web_audit["status"] == "BLOCKED" and not web_audit["model_verified"])
        check("unknown-record-not-found", call("/api/v1/records/" + "0" * 32)[0] == 404)
        self_test_body = '{"backend":"cpu","gpu":-1}'
        check('self-test-requires-session', call('/api/v1/engine/self-test', self_test_body,
              {'Content-Type': 'application/json'})[0] == 403)
        check('self-test-rejects-foreign-origin', call('/api/v1/engine/self-test', self_test_body,
              {**auth, 'Origin': 'https://example.org'})[0] == 403)
        for name, body in [('fractional-gpu', '{"backend":"cpu","gpu":0.5}'),
                           ('boolean-gpu', '{"backend":"cpu","gpu":false}'),
                           ('unknown-backend', '{"backend":"cuda","gpu":0}'),
                           ('oversized-gpu', '{"backend":"vulkan","gpu":65}'),
                           ('duplicate-backend', '{"backend":"cpu","backend":"cpu","gpu":0}'),
                           ('nested-request', '{"backend":"cpu","gpu":{"value":0}}'),
                           ('arbitrary-path', '{"backend":"cpu","gpu":0,"path":"/etc/passwd"}')]:
            code, body, _ = call('/api/v1/engine/self-test', body, auth)
            check('self-test-'+name, code == 422 and json.loads(body)['error']['code'] == 'SELF_TEST_REQUEST')
        for backend in ['cpu'] + (['vulkan'] if args.gpu is not None else []):
            status, body, _ = call('/api/v1/engine/self-test',
                                  json.dumps(dict(backend=backend, gpu=args.gpu if backend == 'vulkan' else -1)), auth)
            diagnostic = json.loads(body)
            check(backend+'-actual-awa-self-test', status == 200 and diagnostic['passed']
                  and diagnostic['status'] == 'PASS' and not diagnostic['model_verified']
                  and len(diagnostic['cases']) == 2
                  and all(c['passed'] and c['execution']['cpu_calls'] == int(backend == 'cpu')
                          and c['execution']['vulkan_calls'] == int(backend == 'vulkan')
                          for c in diagnostic['cases']))
            check(backend+'-self-test-visible-in-history',
                  json.loads(call('/api/v1/records?kind=operator-test')[1])['items'][0]['id'] == diagnostic['record_id'])
            saved_diagnostic = json.loads(call('/api/v1/records/'+diagnostic['record_id'])[1])
            check(backend+'-self-test-retains-raw-report',
                  saved_diagnostic == {k: v for k, v in diagnostic.items() if k != 'record_id'})
        if args.gpu is not None:
            status, body, _ = call('/api/v1/engine/devices')
            check('real-device-probe', status == 200 and any(d['index'] == args.gpu for d in json.loads(body)['devices']))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(lambda _: call("/api/v1/plan", request_text, auth), range(8)))
        check("concurrent-plan-isolation", all(status == 200 and json.loads(body) == expected for status, body, _ in replies))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        errors = process.stderr.read()
        temporary.cleanup()
        if any(marker in errors for marker in ["AddressSanitizer", "UndefinedBehaviorSanitizer", "runtime error:"]):
            raise AssertionError("Sanitizer finding in local Web host: " + errors[:2000])
    report = {"scope": "Drogon host, CLI parity, v1-to-v3 SQLite migration, actual AWA self-tests and HTTP boundary; full image jobs are tested separately by check_jobs.py",
              "platform": "Linux x86_64", "cases": results, "passed": len(results), "status": "PASS"}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
