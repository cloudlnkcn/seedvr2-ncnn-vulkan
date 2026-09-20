#!/usr/bin/env python3
"""Embed compiled SPIR-V words without runtime shader file dependencies."""
import os
import struct
import sys
from pathlib import Path

out = Path(sys.argv[1])
parts = ['#pragma once\n#include <cstdint>\nnamespace seedvr2::engine::shaders {\n']
for source in sys.argv[2:]:
    p = Path(source)
    data = p.read_bytes()
    if len(data) % 4 or data[:4] != b'\x03\x02\x23\x07':
        raise SystemExit(f'Invalid SPIR-V: {p}')
    words = struct.unpack('<'+'I'*(len(data)//4), data)
    name = p.stem.replace('.', '_')
    parts.append(f'inline constexpr uint32_t {name}[] = {{\n')
    parts.extend(','.join(f'0x{x:08x}' for x in words[i:i+8])+',\n' for i in range(0,len(words),8))
    parts.append('};\n')
parts.append('}\n')
out.parent.mkdir(parents=True, exist_ok=True)
# Atomic write: MSBuild runs this custom command from several projects in
# parallel, and a racing writer must never clobber a file being read.
tmp = out.with_suffix(out.suffix + f'.tmp{os.getpid()}')
tmp.write_text(''.join(parts))
os.replace(tmp, out)
