#!/usr/bin/env python3
"""Embed hashed, pnnx-exported smoke fixtures for offline CLI/Web diagnostics."""
import hashlib
import json
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
manifest = json.loads((source/'provenance.json').read_text())
files = {'provenance.json': (source/'provenance.json').read_bytes()}
for item in manifest['cases']:
    path = source/item['path']
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != item['sha256']:
        raise SystemExit('Bundled fixture manifest mismatch')
    files[item['path']] = data
    case = json.loads(data)
    for role in ['model_param', 'model_bin', 'video_qkv', 'text_qkv', 'reference_video', 'reference_text']:
        row = case[role]
        path = Path(item['path']).parent/row['path']
        data = (source/path).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise SystemExit(f'Bundled fixture hash mismatch: {path}')
        files[str(path)] = data
lines = ['// Generated from independently checked fixture bytes.', '#include "fixture.hpp"',
         'namespace seedvr2::engine { namespace {']
for i, (name, data) in enumerate(files.items()):
    lines.append(f'const unsigned char data_{i}[] = {{')
    for start in range(0, len(data), 32):
        lines.append(','.join(str(x) for x in data[start:start+32])+',')
    lines.append('};')
lines += ['} const std::vector<FixtureFile>& awa_fixture() {',
          'static const std::vector<FixtureFile> files = {']
for i, name in enumerate(files):
    lines.append('{'+json.dumps(name)+f', std::string_view(reinterpret_cast<const char*>(data_{i}), sizeof(data_{i}))'+'},')
lines += ['}; return files; } }']
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('\n'.join(lines)+'\n')
