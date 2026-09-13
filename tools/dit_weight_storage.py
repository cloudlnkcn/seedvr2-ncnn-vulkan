#!/usr/bin/env python3
"""Strict storage-only rewriting of this project's lowered FP32 DiT graphs.

Preserve every untagged constant and custom AWA attribute byte. Only standard
InnerProduct matrices receive the ncnn FP16 tag; inference still loads FP32.
Unknown layers, layouts, nonfinite weights and trailing bytes fail closed.
"""
import struct
import os
import tempfile
from pathlib import Path
import numpy as np

NO_WEIGHTS = {'Input', 'Split', 'Reshape', 'Slice', 'BinaryOp', 'Swish'}


def rewrite(param, source, destination):
    if destination.exists():
        raise ValueError('Refusing to overwrite a candidate')
    fd, name = tempfile.mkstemp(prefix='.storage-', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        audit = _rewrite(param, source, temporary)
        os.link(temporary, destination)  # Atomic no-overwrite publication.
        return audit
    finally:
        temporary.unlink(missing_ok=True)


def _rewrite(param, source, destination):
    rows = param.read_text().splitlines()
    if rows[0] != '7767517' or int(rows[1].split()[0]) != len(rows)-2:
        raise ValueError('Unexpected graph header')
    audit = []
    with source.open('rb') as reader, destination.open('wb') as writer:
        def take(size):
            data = reader.read(size)
            if len(data) != size:
                raise ValueError('Truncated weight stream')
            return data

        for row in rows[2:]:
            parts = row.split()
            op, name = parts[:2]
            start = 4 + int(parts[2]) + int(parts[3])
            params = dict(p.split('=', 1) for p in parts[start:])
            if op in NO_WEIGHTS:
                continue
            if op == 'SeedVR2Constant' and params == {'0': '2560'}:
                writer.write(take(2560*4))
            elif op == 'RMSNorm' and params.get('2') == '0':
                continue
            elif op == 'SeedVR2AWA' and params.get('3') == '2' and params.get('0') == '20':
                writer.write(take((1+512+21+3)*4))
            elif op == 'InnerProduct':
                if set(params) - {'0', '1', '2'}:
                    raise ValueError('Unsupported InnerProduct parameters')
                count, outputs, bias = int(params['2']), int(params['0']), int(params['1'])
                if count <= 0 or outputs <= 0 or count % outputs or bias not in (0, 1):
                    raise ValueError('Invalid InnerProduct shape')
                if take(4) != b'\0'*4:
                    raise ValueError('Expected FP32 matrix tag')
                values = np.frombuffer(take(count*4), dtype='<f4')
                if not np.isfinite(values).all() or np.max(np.abs(values)) > 65504:
                    raise ValueError('Weights outside finite FP16 range')
                half = values.astype('<f2')  # IEEE round-to-nearest-even; retain subnormals.
                restored = half.astype('<f4')
                delta = np.abs(values-restored)
                tiny = (np.abs(values) < 2**-14) & (values != 0)
                audit.append(dict(layer=name, elements=count,
                                  changed=int(np.count_nonzero(values != restored)),
                                  max_abs=float(delta.max()),
                                  subnormal_input_count=int(np.count_nonzero(tiny)),
                                  rounded_to_zero=int(np.count_nonzero((values != 0) & (restored == 0)))))
                writer.write(struct.pack('<I', 0x01306B47))
                writer.write(half.tobytes())
                writer.write(b'\0' * ((-count*2) % 4))
                if bias:
                    writer.write(take(outputs*4))
            else:
                raise ValueError(f'Unsupported weight layout: {op} {name}')
        if reader.read(1):
            raise ValueError('Unconsumed weight bytes')
    if not audit:
        raise ValueError('No matrix weights')
    return audit
