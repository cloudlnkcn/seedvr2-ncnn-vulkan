import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import pipeline_contract as contract


def baseline(kind='video'):
    return {'schema_version': f'seedvr2-{kind}-run-v1', 'status': 'SUCCEEDED',
            'profile': f'seedvr2-3b-{kind}-fp32-b-v1',
            'sampling': {'cfg': 1, 'color_fix': 'none', 'latent_scale': .9152, 'steps': 1, 'timestep': 1000},
            'output': {'width': 128, 'height': 128, 'frames': 17},
            'clip': {'decoded_frames': 17, 'padded_frames': 17, 'latent_frames': 5},
            'model_manifest_sha256': 'a'*64}


def reference():
    return json.loads((ROOT/'docs/video-reference-seventeen.json').read_text())


class PipelineContractTest(unittest.TestCase):
    def test_real_reference_has_all_boundaries(self):
        shapes = contract.reference_contract(reference(), baseline(), 'video')
        self.assertEqual(len(shapes), 73)
        self.assertEqual(shapes['block-31-video'], [320, 2560])
        self.assertEqual(shapes['decoded'], [3, 17, 128, 128])

    def test_empty_missing_and_only_passing_are_rejected(self):
        original = reference()
        for stages in ([], original['stages'][:-1], [x for x in original['stages'] if x['passed']]):
            with self.subTest(count=len(stages)), self.assertRaises(ValueError):
                changed = copy.deepcopy(original); changed['stages'] = stages
                contract.reference_contract(changed, baseline(), 'video')

    def test_each_single_boundary_is_required(self):
        original = reference()
        for index in range(73):
            changed = copy.deepcopy(original); changed['stages'].pop(index)
            with self.subTest(index=index), self.assertRaises(ValueError):
                contract.reference_contract(changed, baseline(), 'video')

    def test_duplicate_unknown_and_suffix_are_rejected(self):
        for name in ('prepared', 'other', 'decoded-copy'):
            changed = reference(); changed['stages'][-1]['stage'] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                contract.reference_contract(changed, baseline(), 'video')

    def test_dtype_shape_boolean_dimension_and_path_are_rejected(self):
        for key, value in [('dtype', 'f16le'), ('shape', [3, 17, 64, 128]),
                           ('shape', [True, 17, 128, 128]), ('path', '../prepared.f32'),
                           ('path', 'different.f32'), ('sha256', 'x'*64)]:
            changed = reference(); changed['stages'][0]['reference'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                contract.reference_contract(changed, baseline(), 'video')

    def test_changed_tolerance_and_model_are_rejected(self):
        changed = reference(); changed['tolerance']['atol'] = .01
        with self.assertRaises(ValueError): contract.reference_contract(changed, baseline(), 'video')
        changed_run = baseline(); changed_run['profile'] = 'seedvr2-7b-video-fp32-b-v1'
        with self.assertRaises(ValueError): contract.reference_contract(reference(), changed_run, 'video')

    def test_temporal_padding_and_frame_count_are_required(self):
        for field in ('decoded_frames', 'padded_frames', 'latent_frames'):
            changed = baseline(); changed['clip'][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.reference_contract(reference(), changed, 'video')

    def test_whole_reference_and_baseline_identity_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); bp = root/'run.json'; rp = root/'report.json'; registry = root/'registry.json'
            run = baseline(); bp.write_text(json.dumps(run)); ref = reference(); ref['native_run_sha256'] = contract.sha(bp)
            rp.write_text(json.dumps(ref))
            known = {'kind': 'video', 'baseline_run_sha256': contract.sha(bp), 'model_manifest_sha256': 'a'*64}
            registry.write_text(json.dumps({'references': {contract.sha(rp): known}}))
            with patch.object(contract, 'REGISTRY', registry):
                contract.bind_reference(root, root, 'video')
                ref['scope'] = 'self-signed replacement'; rp.write_text(json.dumps(ref))
                with self.assertRaises(ValueError): contract.bind_reference(root, root, 'video')
                ref.pop('scope'); rp.write_text(json.dumps(ref))
                with self.assertRaises(ValueError): contract.bind_reference(root, root, 'video')

    def test_duplicate_json_and_nonfinite_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input.json'
            for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
                path.write_text(value)
                with self.subTest(value=value), self.assertRaises(ValueError): contract.load_json(path)

    def test_missing_and_changed_tensor_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); p = root/'input.f32'; p.write_bytes(bytes(16))
            row = {'path': 'input.f32', 'shape': [4], 'sha256': contract.sha(p)}
            contract.tensor_path(root, row)
            p.write_bytes(bytes(12))
            with self.assertRaises(ValueError): contract.tensor_path(root, row)
            p.write_bytes(b'\1'*16)
            with self.assertRaises(ValueError): contract.tensor_path(root, row)
            p.unlink()
            with self.assertRaises(ValueError): contract.tensor_path(root, row)

    def test_image_full_reference_uses_same_contract(self):
        ref = json.loads((ROOT/'docs/image-reference-128.json').read_text())
        run = baseline('image'); run['output']['height'] = 80
        self.assertEqual(len(contract.reference_contract(ref, run, 'image')), 73)
        ref['stages'] = []
        with self.assertRaises(ValueError): contract.reference_contract(ref, run, 'image')


if __name__ == '__main__': unittest.main()
