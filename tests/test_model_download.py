"""Exercise actual HTTP resume/integrity behavior with small, local fixture bytes."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_models import download_file

DATA = b'SeedVR2 download contract\n' * 1024
ROW = {'size': len(DATA), 'lfs': {'sha256': hashlib.sha256(DATA).hexdigest()}}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        offset = int(self.headers.get('Range', 'bytes=0-')[6:-1])
        resume = 'Range' in self.headers
        self.send_response(206 if resume and self.path != '/ignore' else 200)
        if resume:
            start = offset + 1 if self.path == '/bad-range' else offset
            self.send_header('Content-Range', f'bytes {start}-{len(DATA)-1}/{len(DATA)}')
        payload = DATA[offset:] if self.path != '/corrupt' else b'X' * (len(DATA) - offset)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


class DownloadContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='模型 download ')
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name) / 'checkpoint.pth'
        self.partial = self.target.with_suffix('.pth.partial')

    def test_download_and_cached_file_never_redownloads(self):
        self.assertEqual(download_file(self.target, ROW, self.url), 'downloaded')
        self.assertEqual(self.target.read_bytes(), DATA)
        self.assertFalse(self.partial.exists())
        with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network')):
            self.assertEqual(download_file(self.target, ROW, self.url), 'cached')

    def test_resume(self):
        self.partial.write_bytes(DATA[:997])
        download_file(self.target, ROW, self.url)
        self.assertEqual(self.target.read_bytes(), DATA)

    def test_completed_partial_promoted_without_request(self):
        self.partial.write_bytes(DATA)
        with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network')):
            self.assertEqual(download_file(self.target, ROW, self.url), 'completed-partial')
        self.assertEqual(self.target.read_bytes(), DATA)

    def test_wrong_resume_range_keeps_partial(self):
        for path in ('/ignore', '/bad-range'):
            with self.subTest(path=path):
                self.partial.write_bytes(DATA[:997])
                with self.assertRaisesRegex(ValueError, 'resume range'):
                    download_file(self.target, ROW, self.url + path)
                self.assertEqual(self.partial.read_bytes(), DATA[:997])
                self.assertFalse(self.target.exists())

    def test_corruption_never_becomes_final_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            download_file(self.target, ROW, self.url + '/corrupt')
        self.assertFalse(self.target.exists())

    def test_existing_wrong_file_is_not_overwritten(self):
        self.target.write_bytes(b'keep me')
        with self.assertRaisesRegex(ValueError, 'Refusing to replace'):
            download_file(self.target, ROW, self.url)
        self.assertEqual(self.target.read_bytes(), b'keep me')

    def test_offline_miss_does_not_use_network(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network')):
            with self.assertRaisesRegex(ValueError, 'Offline checkpoint is missing'):
                download_file(self.target, ROW, self.url, offline=True)
        self.assertFalse(self.partial.exists())

    def test_oversized_partial_is_rejected(self):
        self.partial.write_bytes(DATA + b'!')
        with self.assertRaisesRegex(ValueError, 'exceeds locked size'):
            download_file(self.target, ROW, self.url)

    def test_partial_symlink_is_rejected(self):
        other = self.target.parent / 'other'
        other.write_bytes(DATA[:997])
        self.partial.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'symbolic link'):
            download_file(self.target, ROW, self.url)
        self.assertEqual(other.read_bytes(), DATA[:997])


if __name__ == '__main__':
    unittest.main()
