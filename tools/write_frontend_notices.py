#!/usr/bin/env python3
"""Collect installed production dependency licenses for the embedded Web assets."""
import json
import hashlib
from pathlib import Path
root = Path(__file__).resolve().parents[1]
studio = root / 'apps/studio'
lock = json.loads((studio / 'package-lock.json').read_text())
sections = ['SeedVR2 Studio third-party notices\nGenerated from the exact npm lock and installed production dependencies.\n']
missing = []
for relative, entry in sorted(lock['packages'].items()):
    if not relative or entry.get('dev') or entry.get('optional'):
        continue
    folder = studio / relative
    package = json.loads((folder / 'package.json').read_text())
    licenses = [p for p in folder.iterdir() if p.is_file() and p.name.lower().startswith(('license','licence','notice','copying'))]
    sections.append(f"\n{'='*72}\n{package['name']} {package['version']}\nLicense: {entry.get('license', package.get('license', 'See text'))}\n")
    if not licenses:
        readme = folder / 'README.md'
        if readme.exists() and '## License\n' in readme.read_text():
            sections.append(readme.read_text().split('## License\n', 1)[1])
        else:
            sources = json.loads((root / 'third_party/studio/license-sources.json').read_text())
            source = next((s for s in sources if s['name'] == package['name'] and s['version'] == package['version']), None)
            if source:
                data = (root / source['path']).read_bytes()
                if hashlib.sha256(data).hexdigest() != source['sha256']:
                    raise SystemExit('Supplemental license hash mismatch')
                sections.append('Source: ' + source['url'] + '\n' + data.decode())
            else:
                missing.append(package['name'])
    for path in sorted(licenses):
        sections.append(path.read_text(errors='replace'))
if missing:
    raise SystemExit('Missing production license files: ' + ', '.join(missing))
output = root / 'third_party/studio/NOTICES.txt'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('\n'.join(sections))
print(output)
