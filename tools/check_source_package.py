#!/usr/bin/env python3
"""Check public entry points and retained source identities without model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ENTRY_DOCS = ['README.md', 'README.en.md', 'CONTRIBUTING.md', 'docs/TUTORIAL.md',
              'docs/FIRST-RUN.md', 'docs/ARCHITECTURE.md', 'docs/LICENSING.md',
              'docs/wiki/README.md', 'docs/distribution/README.md']


def check(root):
    results = []

    def record(name, passed, detail):
        results.append(dict(check=name, passed=bool(passed), detail=detail))

    for name in ['LICENSE', 'NOTICE', '.github/workflows/native.yml',
                 '.github/ISSUE_TEMPLATE/bug_report.yml', *ENTRY_DOCS]:
        record('required-file', (root/name).is_file(), name)
    for name in ENTRY_DOCS:
        path = root/name
        if not path.is_file():
            continue
        for href in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in href or href.startswith('#'):
                continue
            target = unquote(href.split('#')[0])
            record('entry-document-local-link', (path.parent/target).exists(), f'{name}: {href}')
    reference = root/'tests/reference/seedvr'
    registry = json.loads((reference/'sources.json').read_text())
    for row in registry['files']:
        path = reference/row['path']
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        record('official-source-sha256', actual == row['sha256'], str(path.relative_to(root)))
    for row in json.loads((root/'dependencies.lock.json').read_text())['vendored']:
        path = root/row['path']
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        record('vendored-source-sha256', actual == row['sha256'], row['path'])
    files = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    forbidden = ('.cache/', '.deps/', '.venv', 'build/', 'dist/')
    for name in filter(None, files):
        path = root/name
        if (name.startswith(forbidden) or '/node_modules/' in name
                or path.suffix in ('.pth', '.pt', '.safetensors', '.sqlite3')):
            record('excluded-private-or-large-artifact', False, name)
        if path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
            record('source-file-size', False, name)
    record('tracked-package-inventory', True, f'{len(list(filter(None, files)))} tracked files inspected')
    return dict(schema_version='seedvr2-source-package-check-v1',
                scope='Entry-document local paths, pinned reference/vendor hashes and tracked-file inventory; '
                      'not external link verification, a secret audit, model or binary validation',
                passed=all(row['passed'] for row in results), checks=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = check(ROOT)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    for row in report['checks']:
        if not row['passed']:
            print('FAIL', row['check'], row['detail'])
    print(('PASS' if report['passed'] else 'FAIL') + f": {len(report['checks'])} source-package checks")
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
