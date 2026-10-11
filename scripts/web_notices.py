#!/usr/bin/env python3
"""Collect redistribution notices from the resolved browser dependencies."""
import hashlib
import json
from pathlib import Path
import subprocess

metadata = json.loads(subprocess.check_output([
    'cargo', 'metadata', '--locked', '--format-version', '1',
    '--filter-platform', 'wasm32-unknown-unknown', '--features', 'web',
], text=True))
nodes = {n['id']: n for n in metadata['resolve']['nodes']}
reachable = set()
queue = [metadata['resolve']['root']]
while queue:
    identity = queue.pop()
    if identity in reachable:
        continue
    reachable.add(identity)
    queue.extend(nodes[identity]['dependencies'])
groups = {}
index = []
for package in sorted(metadata['packages'], key=lambda p: (p['name'], p['version'])):
    if package['id'] not in reachable or not package['source']:
        continue
    base = Path(package['manifest_path']).parent
    files = {p for pattern in ('LICENSE*', 'LICENCE*', 'COPYING*', 'NOTICE*', 'COPYRIGHT*') for p in base.glob(pattern) if p.is_file()}
    if package.get('license_file'):
        files.add(base / package['license_file'])
    # Include nested notices shipped with embedded font/image/shader assets.
    files.update(p for p in base.rglob('*') if p.is_file() and p.name.upper().startswith(('LICENSE', 'LICENCE', 'NOTICE')))
    references = []
    for file in sorted(files):
        try:
            text = file.read_text()
        except UnicodeDecodeError:
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in groups:
            groups[digest] = (len(groups) + 1, text)
        references.append(str(groups[digest][0]))
    index.append(f"{package['name']} {package['version']} — {package.get('license') or 'see notices'} — texts: {', '.join(references) or 'see package source'}")
output = Path('dist/third-party-notices.txt')
output.parent.mkdir(exist_ok=True)
output.write_text('Third-party notices for the browser build\n\n'
    'This document concerns bundled third-party software and assets. It does not set a license for the project author\'s code.\n\n'
    + '\n'.join(index) + '\n\n'
    + '\n'.join(f'=== License text {number} ===\n{text}\n' for number, text in groups.values())
    + '\n=== Embedded Fira Mono font ===\n' + Path('web/FONT-LICENSE.txt').read_text()
    + '\n=== Animal models ===\n' + Path('assets/models/ATTRIBUTION.txt').read_text())
print(f'Bundled notices: {len(index)} resolved packages, {len(groups)} license texts')
