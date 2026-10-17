"""Isolate torchvision's bundled JPEG library from Conda's newer libjpeg.

Run with each target environment's Python. This patches only ELF SONAME,
preserves an original copy, and updates the installed wheel's RECORD hash.
Re-run after reinstalling torchvision 0.23.0+cu128.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    prefix = Path(sys.prefix)
    assert prefix.parent == Path('/mnt/d/anaconda3/envs')
    assert prefix.name in {'torch', 'generate'}
    dist = importlib.metadata.distribution('torchvision')
    assert dist.version == '0.23.0+cu128', dist.version
    files = [p for p in dist.files if str(p).startswith('torchvision.libs/libjpeg.')]
    assert len(files) == 1, files
    relative = files[0]
    library = Path(dist.locate_file(relative))
    expected = library.name
    soname = subprocess.check_output(['patchelf', '--print-soname', str(library)], text=True).strip()
    assert soname in {'libjpeg.so.8', expected}, soname
    image = dist.locate_file('torchvision/image.so')
    needed = subprocess.check_output(['patchelf', '--print-needed', str(image)], text=True).splitlines()
    assert expected in needed, needed
    record_candidates = [p for p in dist.files if str(p).endswith('.dist-info/RECORD')]
    assert len(record_candidates) == 1
    record = Path(dist.locate_file(record_candidates[0]))
    backup = Path('/mnt/d/migration/backups/torchvision-jpeg') / prefix.name
    backup.mkdir(parents=True, exist_ok=True)
    report = dict(prefix=str(prefix), torchvision=dist.version, library=str(library),
                  image_needed=needed, soname_before=soname, sha256_before=sha256(library))
    if soname != expected:
        if (backup / library.name).exists():
            assert sha256(backup / library.name) == sha256(library), 'Original library changed'
        else:
            shutil.copy2(library, backup / library.name)
            shutil.copy2(record, backup / 'RECORD.original')
        # Atomic replacement also prevents modifying an inode shared by Conda cloning.
        with tempfile.TemporaryDirectory(dir=library.parent) as temp:
            patched = Path(temp) / library.name
            shutil.copy2(library, patched)
            subprocess.run(['patchelf', '--set-soname', expected, str(patched)], check=True)
            os.replace(patched, library)
        report['action'] = 'patched_soname'
    else:
        report['action'] = 'already_patched'
    rows = list(csv.reader(io.StringIO(record.read_text())))
    matches = [row for row in rows if row[0] == str(relative)]
    assert len(matches) == 1
    digest = hashlib.sha256(library.read_bytes()).digest()
    matches[0][1] = 'sha256=' + base64.urlsafe_b64encode(digest).decode().rstrip('=')
    matches[0][2] = str(library.stat().st_size)
    output = io.StringIO()
    csv.writer(output, lineterminator='\n').writerows(rows)
    record.write_text(output.getvalue())
    report.update(soname_after=subprocess.check_output(
        ['patchelf', '--print-soname', str(library)], text=True).strip(),
        sha256_after=sha256(library), original_backup=str(backup / library.name),
        original_record=str(backup / 'RECORD.original'), record_updated=True)
    result = Path('/mnt/d/migration/logs') / f'{prefix.name}-jpeg-soname-patch.json'
    # Preserve the original before/after evidence on an idempotent second run.
    if report['action'] != 'already_patched' or not result.exists():
        result.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
