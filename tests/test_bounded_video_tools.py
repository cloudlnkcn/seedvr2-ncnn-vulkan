import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from run_bounded_video_validation import snapshot_native


class NativeSnapshotTest(unittest.TestCase):
    def test_lib_and_lib64_copy_only_seedvr2_and_freeze_symlink_payload(self):
        for name in ('lib', 'lib64'):
            with self.subTest(layout=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                binary = root/'original/bin/seedvr2'
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b'cli')
                library = root/'original'/name
                library.mkdir()
                (library/'libseedvr2.so.0.7.0').write_bytes(b'sdk')
                (library/'libseedvr2.so.0.7').symlink_to('libseedvr2.so.0.7.0')
                (library/'unrelated.dat').write_bytes(b'not part of this model runtime')
                native = snapshot_native(binary, root/'frozen')
                self.assertEqual(native.read_bytes(), b'cli')
                frozen_library = root/'frozen'/name
                self.assertFalse((frozen_library/'unrelated.dat').exists())
                self.assertFalse((frozen_library/'libseedvr2.so.0.7').is_symlink())
                (library/'libseedvr2.so.0.7.0').write_bytes(b'changed later')
                self.assertEqual((frozen_library/'libseedvr2.so.0.7').read_bytes(), b'sdk')

    def test_missing_sdk_is_rejected_before_model_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'original/bin/seedvr2'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'cli')
            with self.assertRaisesRegex(ValueError, 'SDK'):
                snapshot_native(binary, root/'frozen')


if __name__ == '__main__':
    unittest.main()
