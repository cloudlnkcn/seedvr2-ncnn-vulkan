import sys
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from measure_video_quality import pixels, temporal_errors


class VideoQualityTest(unittest.TestCase):
    def test_fixed_fixture_identity_and_target_frame_counts(self):
        root = Path(__file__).resolve().parent/'fixtures/video-bounded'
        manifest = root/'manifest.json'
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(),
                         '45ceb036bd5546304d44bdf3d50d0bb85d55bbd531fc248851b3c8d62f4b47e8')
        for case in json.loads(manifest.read_text())['cases']:
            with self.subTest(case=case['case_id']):
                self.assertEqual(case['target']['shape'], [case['frames'], 80, 128, 3])
                for key in ('input', 'target'):
                    path = root/case[key]['path']
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case[key]['sha256'])
                self.assertEqual((root/case['target']['path']).stat().st_size,
                                 case['frames']*80*128*3)

    def test_empty_video_cannot_produce_a_quality_result(self):
        empty = np.zeros((0, 12, 12, 3), np.uint8)
        with self.assertRaises(ValueError):
            temporal_errors(empty, empty)

    def test_constant_bias_is_distinct_from_changing_error(self):
        target = np.zeros((3, 12, 12, 3), np.float32)
        stable = target+10
        self.assertEqual(temporal_errors(stable, target)['continuous_mean_mae'], 0)
        stable[1] = 30
        self.assertEqual(temporal_errors(stable, target)['continuous_mean_mae'], 20)

    def test_cut_is_retained_and_excluded_from_continuous_statistic(self):
        target = np.zeros((3, 12, 12, 3), np.float32)
        result = target.copy()
        result[1:] = 40
        report = temporal_errors(result, target, cut_before=1)
        self.assertEqual(report['cut_mae'], 40)
        self.assertEqual(report['continuous_mean_mae'], 0)
        self.assertEqual(report['continuous_transition_count'], 1)
        self.assertEqual(len(report['transitions']), 2)

    def test_nonfinite_shape_and_invalid_cut_rejected(self):
        valid = np.zeros((3, 12, 12, 3), np.float32)
        bad = valid.copy(); bad[0, 0, 0, 0] = np.nan
        for source, target, cut in [(bad, valid, None), (valid[:2], valid, None),
                                    (valid, valid, 0), (valid, valid, 3)]:
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                temporal_errors(source, target, cut)

    def test_padded_frames_do_not_enter_rgb_output(self):
        source = np.zeros((3, 5, 12, 12), np.float32)
        source[:, 3:] = 1
        actual = pixels(source, 3)
        self.assertEqual(actual.shape, (3, 12, 12, 3))
        self.assertTrue(np.all(actual == 128))
        with self.assertRaises(ValueError):
            pixels(source, 6)


if __name__ == '__main__':
    unittest.main()
