"""SeedVR2 3B FP32 diagnostic contracts, independent of reported pass counts.

The historical tolerances are development diagnostics, not quality certification.
Whole-report digests authorize retained references; a self-consistent report does
not establish official provenance. New entries require source/output review.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'tests/reference/reviewed-pipelines.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Unbounded report')
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def tolerance(kind):
    result = {'atol': .001, 'rtol': .001, 'calibration': 'DIAGNOSTIC_NOT_MODEL_CERTIFICATION'}
    if kind == 'image':
        result['max_uint8_error'] = 1
    elif kind != 'video':
        raise ValueError('Unknown pipeline kind')
    return result


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Invalid integer in pipeline geometry')
    return value


def expected_shapes(run, kind):
    if run.get('schema_version') != f'seedvr2-{kind}-run-v1' or run.get('status') != 'SUCCEEDED':
        raise ValueError('Expected a complete native run')
    if run.get('profile') != f'seedvr2-3b-{kind}-fp32-b-v1':
        raise ValueError('Unreviewed model profile')
    if run.get('sampling') != {'cfg': 1, 'color_fix': 'none', 'latent_scale': .9152, 'steps': 1, 'timestep': 1000}:
        raise ValueError('Different sampling contract')
    h = integer(run['output']['height'], 64, 512)
    w = integer(run['output']['width'], 64, 512)
    if h % 16 or w % 16:
        raise ValueError('Output must follow the 3B spatial patch contract')
    t, lt = 1, 1
    if kind == 'video':
        clip = run['clip']
        frames = integer(clip['decoded_frames'], 1, 17)
        t = ((frames - 1 + 3) // 4) * 4 + 1
        lt = (t - 1) // 4 + 1
        if (type(clip['padded_frames']) is not int or clip['padded_frames'] != t
                or type(clip['latent_frames']) is not int or clip['latent_frames'] != lt
                or type(run['output']['frames']) is not int or run['output']['frames'] != frames):
            raise ValueError('Temporal padding/latent/crop contract differs')
    spatial = [h // 8, w // 8]
    latent = ([lt] if kind == 'video' else []) + spatial
    decoded = ([t] if kind == 'video' else []) + [h, w]
    tokens = lt * (h // 16) * (w // 16)
    shapes = {'prepared': [3] + decoded, 'posterior': [32] + latent,
              'conditioned': [16] + latent, 'patch-in': [tokens, 2560],
              'text-in': [58, 2560], 'time-in': [2560, 6],
              'velocity': [16] + latent, 'latent': [16] + latent, 'decoded': [3] + decoded}
    for index in range(32):
        shapes[f'block-{index:02d}-video'] = [tokens, 2560]
        shapes[f'block-{index:02d}-text'] = [58, 2560]
    return shapes


def tensor_contract(row, name, shape):
    if (not isinstance(row, dict) or row.get('dtype') != 'f32le'
            or row.get('path') != name + '.f32' or row.get('shape') != shape
            or any(type(x) is not int for x in row.get('shape', []))
            or not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64
            or any(x not in '0123456789abcdef' for x in row['sha256'])):
        raise ValueError('Invalid tensor identity/shape/dtype: ' + name)


def reference_contract(ref, baseline, kind):
    if ref.get('schema_version') != f'seedvr2-{kind}-parity-v1':
        raise ValueError('Reference schema differs')
    if ref.get('tolerance') != tolerance(kind):
        raise ValueError('Historical diagnostic tolerance changed')
    expected = expected_shapes(baseline, kind)
    stages = ref.get('stages')
    if not isinstance(stages, list) or len(stages) != len(expected):
        raise ValueError('Reference must contain all 73 SeedVR2 boundaries')
    seen = set()
    for row in stages:
        name = row.get('stage') if isinstance(row, dict) else None
        if name not in expected or name in seen:
            raise ValueError('Duplicate or unknown pipeline boundary')
        tensor_contract(row.get('reference'), name, expected[name])
        seen.add(name)
    if seen != set(expected):
        raise ValueError('Incomplete pipeline boundaries')
    return expected


def bind_reference(reference_root, baseline_root, kind):
    report_path, baseline_path = Path(reference_root)/'report.json', Path(baseline_root)/'run.json'
    ref, baseline = load_json(report_path), load_json(baseline_path)
    expected = reference_contract(ref, baseline, kind)
    digest = sha(report_path)
    registry = load_json(REGISTRY)
    known = registry['references'].get(digest)
    if not known or known['kind'] != kind:
        raise ValueError('Reference is not in the reviewed whole-report registry')
    if (sha(baseline_path) != known['baseline_run_sha256']
            or ref.get('native_run_sha256') != known['baseline_run_sha256']
            or baseline.get('model_manifest_sha256') != known['model_manifest_sha256']):
        raise ValueError('Reference source/run/model binding differs')
    return ref, baseline, expected, digest, known


def bind_candidate(run, baseline, kind, expected):
    if expected_shapes(run, kind) != expected:
        raise ValueError('Candidate geometry differs')
    for field in ('profile', 'model_manifest_sha256', 'input', 'sampling', 'seed', 'noise_algorithm'):
        if run.get(field) != baseline.get(field):
            raise ValueError('Candidate execution differs: ' + field)
    if kind == 'video' and run.get('clip') != baseline.get('clip'):
        raise ValueError('Candidate temporal execution differs')
    for name, shape in expected.items():
        tensor_contract(run['diagnostics'].get(name), name, shape)
    for name in ('posterior-noise', 'noise'):
        tensor_contract(run['diagnostics'].get(name), name, expected['latent'])
        tensor_contract(baseline['diagnostics'].get(name), name, expected['latent'])


def tensor_path(root, row):
    root = Path(root)
    name = row['path']
    if Path(name).name != name or '\\' in name:
        raise ValueError('Tensor must be a direct local artifact')
    path = root/name
    if path.is_symlink() or not path.is_file() or path.stat().st_size != math.prod(row['shape']) * 4:
        raise ValueError('Missing, indirect or wrong-sized tensor')
    if sha(path) != row['sha256']:
        raise ValueError('Tensor content differs')
    return path
