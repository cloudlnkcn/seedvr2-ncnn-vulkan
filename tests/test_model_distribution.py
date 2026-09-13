"""Delivery failures must not become apparently installed, trusted model packages."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import model_distribution as dist


class DistributionContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='模型 分发 ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / 'original'
        self.model.mkdir()
        payloads = {'encoder/model.param': b'fixture graph', 'encoder/model.bin': b'fixture weights',
                    'text.f32': b'\x00\x00\x80\x3f'}
        refs = {}
        for name, raw in payloads.items():
            path = self.model / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
            refs[name] = dict(path=name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        self.manifest = dict(profile='test-only', model_verified=False,
                             graphs=[dict(id='encoder', param=refs['encoder/model.param'],
                                          weights=refs['encoder/model.bin'])],
                             constants=dict(text=dict(path='text.f32', dtype='f32le', shape=[1],
                                                      sha256=refs['text.f32']['sha256'])))
        (self.model / 'manifest.json').write_text(json.dumps(self.manifest))
        self.reviewed = dict(packages=[dict(kind=kind, profile='test-only',
                                           payload_sha256=dist.identity(self.manifest))
                                      for kind in ('image', 'video')])
        self.bundle = self.root / 'bundle'
        self.output = self.root / '安装 中文'

    def staged(self, both=False):
        packages = dict(image=self.model)
        if both:
            packages['video'] = self.model
        report = dist.stage(packages, self.bundle, self.reviewed)
        self.catalog = json.loads((self.bundle / 'catalog.json').read_text())
        return report

    def install(self, **kwargs):
        return dist.install(self.catalog, 'image', self.output, self.reviewed,
                            source=self.bundle, **kwargs)

    def test_deduplicated_bundle_and_offline_unicode_install(self):
        report = self.staged(both=True)
        self.assertEqual(report['objects'], 4)
        self.assertEqual(report['unique_bytes'] * 2, report['package_bytes'])
        with patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')):
            self.assertTrue(self.install(offline=True)['passed'])
            self.assertTrue(self.install(offline=True)['cached'])
        self.assertEqual((self.output / 'encoder/model.bin').read_bytes(), b'fixture weights')
        self.assertFalse((self.output / '.seedvr2-install.json').exists())

    def test_unreviewed_manifest_cannot_stage_even_if_its_files_hash(self):
        manifest = deepcopy(self.manifest)
        manifest['profile'] = 'unreviewed'
        (self.model / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            self.staged()
        self.assertFalse(self.bundle.exists())

    def test_missing_catalogue_file_cannot_be_silently_omitted(self):
        self.staged()
        self.catalog['packages'][0]['files'].pop()
        with self.assertRaisesRegex(ValueError, 'file contract'):
            self.install()
        self.assertFalse(self.output.exists())

    def test_duplicate_kind_rejected(self):
        self.staged()
        self.catalog['packages'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.install()

    def test_manifest_object_must_match_inline_reviewed_manifest(self):
        self.staged()
        row = self.catalog['packages'][0]['manifest_object']
        raw = b'{"model_verified": true}'
        row.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        (self.bundle / 'objects' / row['sha256']).write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            self.install()
        self.assertFalse(self.output.exists())

    def test_corrupt_object_does_not_create_final_directory(self):
        self.staged()
        row = self.catalog['packages'][0]['files'][0]
        (self.bundle / 'objects' / row['sha256']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            self.install()
        self.assertFalse(self.output.exists())

    def test_missing_object_resumes_from_checked_partial_files(self):
        self.staged()
        row = self.catalog['packages'][0]['files'][-1]
        path = self.bundle / 'objects' / row['sha256']
        raw = path.read_bytes()
        path.unlink()
        with self.assertRaises(ValueError):
            self.install()
        self.assertFalse(self.output.exists())
        partial = self.output.with_name(self.output.name + '.partial')
        self.assertTrue((partial / 'manifest.json').is_file())
        path.write_bytes(raw)
        self.assertTrue(self.install()['passed'])
        self.assertFalse(partial.exists())

    def test_completed_partial_is_recovered(self):
        self.staged()
        self.install()
        self.output.rename(self.output.with_name(self.output.name + '.partial'))
        self.assertTrue(self.install()['completed_partial'])

    def test_corrupt_final_package_is_not_overwritten(self):
        self.staged()
        self.install()
        target = self.output / 'text.f32'
        target.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            self.install()
        self.assertEqual(target.read_bytes(), b'keep')

    def test_unrelated_partial_directory_is_preserved(self):
        self.staged()
        partial = self.output.with_name(self.output.name + '.partial')
        partial.mkdir()
        (partial / 'important').write_text('keep')
        with self.assertRaises(ValueError):
            self.install()
        self.assertEqual((partial / 'important').read_text(), 'keep')

    def test_symbolic_link_in_package_parent_is_rejected(self):
        self.staged()
        parent = self.root / 'elsewhere'
        parent.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(parent, target_is_directory=True)
        self.output = alias / 'model'
        with self.assertRaisesRegex(ValueError, 'symbolic'):
            self.install()
        self.assertFalse(list(parent.iterdir()))

    def test_second_installer_cannot_write_same_destination(self):
        self.staged()
        with dist.install_lock(self.output):
            with self.assertRaisesRegex(ValueError, 'Another installer'):
                self.install()

    def test_plan_is_read_only(self):
        self.staged()
        self.assertTrue(self.install(plan=True)['plan'])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name(self.output.name + '.partial').exists())

    def test_repository_catalogue_is_bound_to_native_reviewed_policy(self):
        catalogue = json.loads((dist.ROOT / 'docs/distribution/catalog.v1.json').read_text())
        policy = json.loads((dist.ROOT / 'policies/reviewed-packages.json').read_text())
        objects = dist.check_catalog(catalogue, policy)
        self.assertTrue(objects)
        catalogue['packages'][0]['manifest']['sampling']['steps'] = 99
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            dist.check_catalog(catalogue, policy)

    def test_hardlink_install_preserves_original_bytes(self):
        self.staged()
        self.install(hardlink=True)
        row = self.catalog['packages'][0]['files'][0]
        self.assertEqual((self.output / row['path']).stat().st_ino,
                         (self.bundle / 'objects' / row['sha256']).stat().st_ino)

    def test_flat_release_download_checks_every_object(self):
        self.staged()
        base = 'https://github.com/example/model/releases/download/fixed-v1'
        urls = []
        def fetch(target, row, url, offline):
            self.assertFalse(offline)
            sha = row['lfs']['sha256']
            self.assertEqual(url, base + '/' + sha)
            urls.append(url)
            target.write_bytes((self.bundle / 'objects' / sha).read_bytes())
        with patch.object(dist, 'download_file', side_effect=fetch):
            result = dist.install(self.catalog, 'image', self.output, self.reviewed,
                                  base_url=base, layout='flat')
        self.assertTrue(result['passed'])
        self.assertEqual(len(urls), 4)
        dist.installed(self.output, self.catalog['packages'][0])

    def test_unknown_layout_cannot_write(self):
        self.staged()
        with self.assertRaisesRegex(ValueError, 'layout'):
            self.install(layout='../')
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
