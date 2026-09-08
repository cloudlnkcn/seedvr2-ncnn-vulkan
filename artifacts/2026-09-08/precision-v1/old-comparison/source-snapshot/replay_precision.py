#!/usr/bin/env python3
"""Recompute native precision-probe statistics from retained raw tensors."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from pipeline_contract import load_json, sha
from check_precision import CASES, MODES


def replay(root):
    matrix = load_json(root/'report.json')
    expected = matrix['expected_cases']
    if (not expected or len(set(expected)) != len(expected)
            or [row['id'] for row in matrix['cases']] != expected
            or any(key not in {a+'/'+b for a in CASES for b in MODES} for key in expected)):
        raise ValueError('Empty, duplicate, unknown or missing precision cases')
    counts = {'erf': 1024, 'celu': 1024, 'reduction': 1, 'reduction-local': 1,
              'reduction-cancellation': 1, 'sdpa': 166400, 'patch-in': 819200}
    identities, rows = {}, []
    for row in matrix['cases']:
        path = root/row['report']
        if path.parent != root or sha(path) != row['report_sha256']:
            raise ValueError('Native report identity differs')
        if sha(root/row['stderr']) != row['stderr_sha256']:
            raise ValueError('Native log identity differs')
        if row['status'] != 'EXECUTED':
            rows.append(dict(id=row['id'], status=row['status'], reason=row['reason']))
            continue
        native = load_json(path)
        case, mode = row['id'].split('/')
        if (native['case'] != case or native['precision'] != mode or native['status'] != 'EXECUTED'
                or native['dispatch'] != 'EXPLICIT_VULKAN_NO_CPU_FALLBACK'
                or native['requested_options'] != native['effective_options']
                or native['tolerance'] != {'atol': .001, 'rtol': .001,
                    'calibration': 'FP32_DIAGNOSTIC_NOT_LOW_PRECISION_CERTIFICATION'}):
            raise ValueError('Native precision/execution contract differs')
        folder = root/(case+'-'+mode)
        def tensor(item, count=None):
            name = item['path']
            if Path(name).name != name or '\\' in name or item['dtype'] != 'f32le':
                raise ValueError('Invalid tensor path/type')
            target = folder/name
            if target.is_symlink() or sha(target) != item['sha256']:
                raise ValueError('Tensor identity differs')
            n = item['elements'] if count is None else count
            if type(n) is not int or n < 1 or item['elements'] != n or target.stat().st_size != n*4:
                raise ValueError('Tensor element count differs')
            return np.fromfile(target, dtype='<f4').astype(np.float64)
        actual, reference = tensor(native['output'], counts[case]), tensor(native['reference'], counts[case])
        if not np.isfinite(reference).all():
            raise ValueError('Non-finite reference')
        for item in native['inputs']:
            if not np.isfinite(tensor(item)).all():
                raise ValueError('Non-finite input')
        identity = tuple(item['sha256'] for item in native['inputs'])+(native['reference']['sha256'],)
        if case in identities and identities[case] != identity:
            raise ValueError('Inputs/reference changed between precisions')
        identities[case] = identity
        finite = np.isfinite(actual)
        error = np.abs(actual-reference)
        budget = .001+.001*np.abs(reference)
        violations = int(np.count_nonzero(~finite | (error > budget)))
        metrics = dict(elements=int(actual.size), nonfinite=int(np.count_nonzero(~finite)),
                       violations=violations, passed=violations == 0,
                       max_abs=float(error.max()) if finite.all() else None,
                       rmse=float(np.sqrt(np.mean(error*error))) if finite.all() else None,
                       max_error_over_limit=float(np.max(error/budget)) if finite.all() else None)
        for key, value in metrics.items():
            reported = native['metrics'][key]
            if isinstance(value, float):
                if not np.isclose(value, reported, rtol=1e-11, atol=1e-15):
                    raise ValueError('Native statistics differ: '+row['id']+' '+key)
            elif value != reported:
                raise ValueError('Native statistics differ: '+row['id']+' '+key)
        if row['metrics'] != native['metrics']:
            raise ValueError('Matrix statistics differ from native report')
        rows.append(dict(id=row['id'], status='RECOMPUTED', metrics=metrics))
    return dict(schema_version='ncnn-precision-replay-v1', source_report_sha256=sha(root/'report.json'),
                script_sha256=sha(Path(__file__)), model_verified=False,
                statistics_verified=any(row['status'] == 'RECOMPUTED' for row in rows), cases=rows,
                recomputed=sum(row['status'] == 'RECOMPUTED' for row in rows),
                fp32_budget_passed=sum(row['status'] == 'RECOMPUTED' and row['metrics']['passed'] for row in rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = replay(args.run)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}, indent=2))
