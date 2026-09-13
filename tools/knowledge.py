#!/usr/bin/env python3
"""Small, source-backed project wiki: ingest evidence, lint links, query reviewed pages.

Ingestion never executes a source or promotes it into a reviewed conclusion.
Only local files and explicitly listed HTTPS documents are supported.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r'[0-9a-f]{64}')
ID = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)*')
LIMIT = 8 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def local_path(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
        raise ValueError(f'Unsafe repository path: {name}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path leaves repository: {name}')
    return path


def read_registry(root):
    data = json.loads((root / 'docs/wiki/sources.json').read_text())
    if data.get('schema_version') != 'seedvr2-knowledge-v1':
        raise ValueError('Unknown knowledge registry version')
    if not data.get('sources') or not data.get('pages'):
        raise ValueError('Empty sources or pages cannot pass lint')
    return data


def lint(root):
    data = read_registry(root)
    errors = []
    sources, pages = {}, {}
    for row in data['sources']:
        sid = row.get('id', '')
        if not ID.fullmatch(sid) or sid in sources:
            errors.append(f'Invalid or duplicate source id: {sid}')
        sources[sid] = row
        for key in ['title', 'url', 'revision', 'checked_at', 'scope']:
            if not isinstance(row.get(key), str) or not row[key].strip():
                errors.append(f'{sid}: missing {key}')
        if not row.get('files'):
            errors.append(f'{sid}: no inspected files')
        for item in row.get('files', []):
            if not SHA.fullmatch(item.get('sha256', '')):
                errors.append(f'{sid}: invalid SHA-256')
            if item.get('path'):
                path = local_path(root, item['path'])
                if not path.is_file() or digest(path.read_bytes()) != item['sha256']:
                    errors.append(f'{sid}: source changed or missing: {item["path"]}; reconcile before updating hash')
            elif not item.get('url', '').startswith('https://'):
                errors.append(f'{sid}: file needs a repository path or HTTPS URL')
    page_paths = set()
    for row in data['pages']:
        pid = row.get('id', '')
        if not ID.fullmatch(pid) or pid in pages:
            errors.append(f'Invalid or duplicate page id: {pid}')
        pages[pid] = row
        path = local_path(root, row['path'])
        if row['path'] in page_paths:
            errors.append(f'{pid}: duplicate page path')
        page_paths.add(row['path'])
        if row.get('kind') not in ('entity', 'concept', 'synthesis'):
            errors.append(f'{pid}: unknown page kind')
        if row.get('status') not in ('reviewed', 'open'):
            errors.append(f'{pid}: unknown review status')
        if not row.get('sources'):
            errors.append(f'{pid}: no source provenance')
        for sid in row.get('sources', []):
            if sid not in sources:
                errors.append(f'{pid}: unknown source: {sid}')
        if not path.is_file():
            errors.append(f'{pid}: missing page: {row["path"]}')
    for pid, row in pages.items():
        if not row.get('related'):
            errors.append(f'{pid}: isolated page')
        for other in row.get('related', []):
            if other not in pages:
                errors.append(f'{pid}: unknown related page: {other}')
            elif pid not in pages[other].get('related', []):
                errors.append(f'{pid}: missing backlink from {other}')
    wiki = root / 'docs/wiki'
    for path in sorted(wiki.rglob('*.md')):
        if path.parent.name in ('entities', 'concepts', 'synthesis') and str(path.relative_to(root)) not in page_paths:
            errors.append(f'Unregistered wiki page: {path.relative_to(root)}')
        for href in re.findall(r'\]\(([^)\s]+)\)', path.read_text()):
            parsed = urlsplit(href)
            if parsed.scheme or href.startswith('#'):
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(root.resolve()) or not target.exists():
                errors.append(f'{path.relative_to(root)}: broken local link: {href}')
    return dict(schema_version='seedvr2-knowledge-lint-v1', passed=not errors,
                sources=len(sources), pages=len(pages), errors=errors,
                scope='Local source hashes, metadata, page coverage and bidirectional links; '
                      'not factual truth, remote freshness, numerical or quality certification')


def ingest(root, source_id, output):
    """Append content-addressed snapshots and a receipt; never rewrite the registry."""
    sources = {x['id']: x for x in read_registry(root)['sources']}
    if source_id not in sources or not ID.fullmatch(source_id):
        raise ValueError(f'Unknown source: {source_id}')
    rows = []
    for item in sources[source_id]['files']:
        if item.get('path'):
            path = local_path(root, item['path'])
            if path.stat().st_size > LIMIT:
                raise ValueError('Evidence is too large; ingest a bounded report instead of weights')
            raw = path.read_bytes()
        else:
            if not item.get('url', '').startswith('https://'):
                raise ValueError('Remote sources must use HTTPS')
            request = urllib.request.Request(item['url'], headers={'User-Agent': 'SeedVR2-knowledge/1'})
            with urllib.request.urlopen(request, timeout=30) as response:
                if not response.geturl().startswith('https://'):
                    raise ValueError('Refusing non-HTTPS redirect')
                raw = response.read(LIMIT + 1)
            if len(raw) > LIMIT:
                raise ValueError('Evidence exceeds 8 MiB; no truncated snapshot is accepted')
        actual = digest(raw)
        folder = output / source_id
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = folder / (actual + '.source')
        if snapshot.exists():
            if digest(snapshot.read_bytes()) != actual:
                raise ValueError(f'Existing snapshot is corrupt: {snapshot}')
        else:
            with snapshot.open('xb') as stream:
                stream.write(raw)
        rows.append(dict(locator=item.get('path') or item['url'], sha256=actual,
                         expected_sha256=item['sha256'], unchanged=actual == item['sha256'],
                         snapshot=str(snapshot), bytes=len(raw)))
    receipt = dict(schema_version='seedvr2-ingest-v1', source=source_id,
                   ingested_at=datetime.now(timezone.utc).isoformat(), files=rows,
                   needs_reconciliation=any(not x['unchanged'] for x in rows),
                   meaning='Captured evidence; no automatic review or source instructions executed')
    path = output / source_id / ('receipt-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    with path.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return receipt


def query(root, phrase):
    terms = phrase.casefold().split()
    if not terms:
        raise ValueError('A nonempty query is required')
    found = []
    for page in read_registry(root)['pages']:
        content = local_path(root, page['path']).read_text()
        score = sum(content.casefold().count(term) for term in terms)
        if score:
            excerpts = [line for line in content.splitlines()
                        if any(term in line.casefold() for term in terms)]
            found.append(dict(page=page['path'], status=page['status'], score=score,
                              sources=page['sources'], related=page['related'], excerpts=excerpts[:4]))
    return dict(query=phrase, method='Literal keyword retrieval, not generated reasoning',
                results=sorted(found, key=lambda x: (-x['score'], x['page'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('lint')
    check.add_argument('--output', type=Path)
    capture = sub.add_parser('ingest')
    capture.add_argument('--source', required=True)
    capture.add_argument('--output', type=Path, default=ROOT / '.cache/knowledge')
    search = sub.add_parser('query')
    search.add_argument('phrase')
    args = parser.parse_args()
    if args.command == 'lint':
        result = lint(ROOT)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    elif args.command == 'ingest':
        result = ingest(ROOT, args.source, args.output)
    else:
        result = query(ROOT, args.phrase)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get('passed') is False or result.get('needs_reconciliation') else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
