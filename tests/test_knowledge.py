"""Source changes and broken provenance must remain visible rather than auto-approved."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import knowledge


class KnowledgeContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'docs/wiki/entities').mkdir(parents=True)
        (self.root / 'source.txt').write_text('known source')
        (self.root / 'docs/wiki/entities/a.md').write_text('# Video cache\n[other](b.md)\n')
        (self.root / 'docs/wiki/entities/b.md').write_text('# Precision\n[other](a.md)\n')
        self.data = dict(schema_version='seedvr2-knowledge-v1', sources=[dict(
            id='source', title='Fixture', url='https://example.org/source', revision='fixture-1',
            checked_at='2026-09-13', scope='Fixture only', files=[dict(path='source.txt',
            sha256=hashlib.sha256(b'known source').hexdigest())])], pages=[
            dict(id='a', path='docs/wiki/entities/a.md', kind='entity', status='reviewed',
                 sources=['source'], related=['b']),
            dict(id='b', path='docs/wiki/entities/b.md', kind='entity', status='open',
                 sources=['source'], related=['a'])])
        self.save()

    def save(self):
        (self.root / 'docs/wiki/sources.json').write_text(json.dumps(self.data))

    def test_linked_evidence_and_query(self):
        self.assertTrue(knowledge.lint(self.root)['passed'])
        result = knowledge.query(self.root, 'cache')['results']
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['sources'], ['source'])

    def test_changed_source_fails_and_ingest_does_not_approve_it(self):
        original = deepcopy(self.data)
        (self.root / 'source.txt').write_text('new conflicting source')
        self.assertFalse(knowledge.lint(self.root)['passed'])
        report = knowledge.ingest(self.root, 'source', self.root / 'snapshots')
        self.assertTrue(report['needs_reconciliation'])
        self.assertEqual(knowledge.read_registry(self.root), original)
        self.assertEqual(Path(report['files'][0]['snapshot']).read_text(), 'new conflicting source')

    def test_snapshot_reuse_does_not_overwrite_previous_evidence(self):
        first = knowledge.ingest(self.root, 'source', self.root / 'snapshots')
        second = knowledge.ingest(self.root, 'source', self.root / 'snapshots')
        self.assertEqual(first['files'][0]['snapshot'], second['files'][0]['snapshot'])
        self.assertEqual(len(list((self.root / 'snapshots/source').glob('receipt-*.json'))), 2)

    def test_unknown_source_and_asymmetric_links_fail(self):
        self.data['pages'][0]['sources'] = ['missing']
        self.data['pages'][1]['related'] = []
        self.save()
        result = knowledge.lint(self.root)
        self.assertFalse(result['passed'])
        self.assertTrue(any('unknown source' in x for x in result['errors']))
        self.assertTrue(any('backlink' in x for x in result['errors']))

    def test_broken_markdown_link_and_unregistered_page_fail(self):
        (self.root / 'docs/wiki/entities/c.md').write_text('[gone](gone.md)')
        result = knowledge.lint(self.root)
        self.assertFalse(result['passed'])
        self.assertTrue(any('Unregistered' in x for x in result['errors']))
        self.assertTrue(any('broken local link' in x for x in result['errors']))

    def test_empty_registry_cannot_pass(self):
        self.data['sources'] = []
        self.save()
        with self.assertRaisesRegex(ValueError, 'Empty'):
            knowledge.lint(self.root)


if __name__ == '__main__':
    unittest.main()
