"""Remote inventory and downloaded-byte gates, without credentials or networking."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))


class ReadbackTests(unittest.TestCase):
    def execute(self, case='valid'):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'source'; source.mkdir()
            (source/'catalog.json').write_text('{}')
            output = root/'download'; output.mkdir()
            downloaded = output/'catalog.json'
            downloaded.write_text('[]' if case == 'corrupt-download' else '{}')
            entries = [SimpleNamespace(rfilename='catalog.json', size=2,
                lfs=SimpleNamespace(sha256='0'*64) if case == 'bad-remote-hash' else None)]
            if case == 'missing': entries = []
            if case == 'extra': entries.append(SimpleNamespace(rfilename='unexpected',size=1,lfs=None))
            if case == 'bad-size': entries[0].size=3
            api = Mock(); api.model_info.return_value = SimpleNamespace(siblings=entries,private=True)
            download = Mock(return_value=str(downloaded))
            fake = SimpleNamespace(HfApi=Mock(return_value=api),hf_hub_download=download)
            with patch.dict(sys.modules, {'huggingface_hub':fake}):
                spec=importlib.util.spec_from_file_location('hf_readback_under_test',ROOT/'tools/verify_hf_model.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            args=['verify','--repo','example/model','--revision','a'*40,'--source',str(source),
                  '--output',str(output),'--report',str(root/'report.json')]
            with patch.object(sys,'argv',args), patch.object(module,'check_catalog'):
                if case == 'valid':
                    module.main(); report=json.loads((root/'report.json').read_text())
                    self.assertTrue(report['passed']);self.assertFalse(report['model_verified'])
                    self.assertEqual(download.call_args.kwargs['revision'],'a'*40)
                else:
                    with self.assertRaises(ValueError): module.main()
                    self.assertFalse((root/'report.json').exists())
    def test_valid_download(self): self.execute()
    def test_missing_remote_file(self): self.execute('missing')
    def test_extra_remote_file(self): self.execute('extra')
    def test_wrong_remote_size(self): self.execute('bad-size')
    def test_wrong_remote_hash(self): self.execute('bad-remote-hash')
    def test_corrupt_download(self): self.execute('corrupt-download')


if __name__ == '__main__': unittest.main()
