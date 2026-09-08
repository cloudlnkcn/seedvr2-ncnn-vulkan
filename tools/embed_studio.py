#!/usr/bin/env python3
"""Embed only Vite production assets, never mount the source/workspace directory."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('dist', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
mime = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
        '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml'}
assets = []
arrays = []
for path in sorted(args.dist.rglob('*')):
    if not path.is_file():
        continue
    if path.suffix not in mime or path.is_symlink():
        raise SystemExit(f'Unexpected production asset: {path}')
    data = path.read_bytes()
    symbol = 'asset_' + str(len(arrays))
    values = ',\n'.join(','.join(str(b) for b in data[i:i+128]) for i in range(0, len(data), 128))
    arrays.append(f'static const unsigned char {symbol}[] = {{\n{values}\n}};')
    route = '/' if path.name == 'index.html' else '/' + path.relative_to(args.dist).as_posix()
    assets.append('{"' + route + '", "' + mime[path.suffix] + '", {reinterpret_cast<const char*>(' + symbol + '), sizeof(' + symbol + ')}}')
if not any(p.name == 'index.html' for p in args.dist.iterdir()):
    raise SystemExit('Build the Studio frontend first')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text('#include "assets.hpp"\nnamespace seedvr2::server {\n' + '\n'.join(arrays) + '\nstatic const Asset assets[] = {\n' + ',\n'.join(assets) + '\n};\nstd::span<const Asset> studio_assets() { return assets; }\n}\n')
