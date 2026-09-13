import hashlib,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import download_models as downloads
from compare_storage_precision import compare


class DownloadRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.old=downloads.ROOT;downloads.ROOT=Path(self.temp.name)
        self.addCleanup(setattr,downloads,'ROOT',self.old)
        self.root=downloads.ROOT/'docs/distribution';self.root.mkdir(parents=True)
        self.catalog=self.root/'catalog.json';self.catalog.write_text('{}')
        self.entry=dict(download_verified=True,revision='a'*40,repo_id='owner/model',catalog='catalog.json',catalog_sha256=hashlib.sha256(b'{}').hexdigest())
    def save(self):
        (self.root/'huggingface-models.json').write_text(json.dumps({'models':{'dit-fp16':self.entry}}))
    def test_pinned_entry(self):
        self.save();entry,catalog=downloads.resolve_entry('dit-fp16');self.assertEqual(catalog,{});self.assertEqual(entry['revision'],'a'*40)
    def test_unverified_rejected(self):
        self.entry['download_verified']=False;self.save()
        with self.assertRaises(ValueError):downloads.resolve_entry('dit-fp16')
    def test_mutable_revision_rejected(self):
        self.entry['revision']='main';self.save()
        with self.assertRaises(ValueError):downloads.resolve_entry('dit-fp16')
    def test_catalog_changed_rejected(self):
        self.save();self.catalog.write_text('{"different":true}')
        with self.assertRaises(ValueError):downloads.resolve_entry('dit-fp16')
    def test_catalog_escape_rejected(self):
        self.entry['catalog']='../catalog.json';self.save()
        with self.assertRaises(ValueError):downloads.resolve_entry('dit-fp16')


class DriftTests(unittest.TestCase):
    def test_loss_is_reported_not_rejected(self):
        result=compare(np.array([2.]),np.array([1.]),{'atol':.001,'rtol':.001})
        self.assertEqual(result['fp32_threshold_violations'],1)
        self.assertEqual(result['rmse'],1.)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):compare(np.array([np.nan]),np.array([1.]),{'atol':.001,'rtol':.001})
    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):compare(np.ones(2),np.ones(1),{'atol':.001,'rtol':.001})

if __name__=='__main__':unittest.main()
