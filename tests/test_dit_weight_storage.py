"""Storage regressions: IEEE tiny values, opaque bytes, malformed graph streams."""
import sys
import tempfile
import unittest
import struct
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from dit_weight_storage import rewrite


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.param, self.source, self.output = [self.root / x for x in ('a.param', 'a.bin', 'b.bin')]
        self.values = np.array([2**-20, -2**-20, 1, 1.001], dtype='<f4')
        self.param.write_text('7767517\n1 2\nInnerProduct linear 1 1 in out 0=2 1=1 2=4\n')
        self.source.write_bytes(bytes(4) + self.values.tobytes() + np.array([3, 4], dtype='<f4').tobytes())

    def test_ieee_and_bias(self):
        audit = rewrite(self.param, self.source, self.output)
        data = self.output.read_bytes()
        self.assertEqual(struct.unpack('<I', data[:4])[0], 0x01306B47)
        actual = np.frombuffer(data[4:12], dtype='<f2').astype('<f4')
        np.testing.assert_array_equal(actual[:2], self.values[:2])
        self.assertEqual(data[12:], self.source.read_bytes()[-8:])
        self.assertEqual(audit[0]['subnormal_input_count'], 2)

    def check_reject(self):
        with self.assertRaises(ValueError):
            rewrite(self.param, self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.storage-*')))

    def test_truncated(self):
        self.source.write_bytes(self.source.read_bytes()[:-1]); self.check_reject()

    def test_trailing(self):
        self.source.write_bytes(self.source.read_bytes()+b'x'); self.check_reject()

    def test_unknown_layer(self):
        self.param.write_text(self.param.read_text().replace('InnerProduct', 'Unreviewed')); self.check_reject()

    def test_quantized_input(self):
        self.source.write_bytes(struct.pack('<I', 0x01306B47)+self.source.read_bytes()[4:]); self.check_reject()

    def test_nonfinite(self):
        self.source.write_bytes(bytes(4)+np.array([float('nan'), 1, 2, 3], dtype='<f4').tobytes()+bytes(8)); self.check_reject()

    def test_no_overwrite(self):
        self.output.write_bytes(b'baseline')
        with self.assertRaises(ValueError): rewrite(self.param, self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b'baseline')


if __name__ == '__main__':
    unittest.main()
