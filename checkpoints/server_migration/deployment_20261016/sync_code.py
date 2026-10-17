#!/usr/bin/env python3
"""Copy tracked project files and Git worktrees without ignored research data."""

import argparse
import datetime as dt
import json
from pathlib import Path
import shlex
import subprocess


ROOT = Path('/mnt/d/python')
RUNTIME = Path('/mnt/d/migration-transfer-logs/code')
SSH = ['ssh', '-S', '/tmp/hola-migration-ssh-%C', '-o', 'BatchMode=yes',
       '-p', '12222', 'root@172.31.224.142']
RSYNC = ['rsync', '-a', '--protect-args', '--partial', '--stats',
         '-e', shlex.join(SSH[:-1])]


def run(args, **kwargs):
    print(shlex.join(map(str, args)), flush=True)
    return subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-thirdparty', action='store_true')
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    records = []
    listing = subprocess.check_output(['git', 'worktree', 'list', '--porcelain'], cwd=ROOT)
    for block in listing.decode().strip().split('\n\n'):
        fields = dict(line.split(' ', 1) for line in block.splitlines() if ' ' in line)
        root = Path(fields['worktree'])
        entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=root)
        files = []
        gitlinks = []
        for entry in entries.split(b'\0'):
            if not entry:
                continue
            metadata, name = entry.split(b'\t', 1)
            if metadata.split()[0] == b'160000':
                gitlinks.append(root / name.decode())
            else:
                if not (root / name.decode()).exists() and not (root / name.decode()).is_symlink():
                    raise RuntimeError(f'Tracked file missing: {root}/{name!r}')
                files.append(name)
        if root != ROOT:
            files.append(b'.git')
        manifest = RUNTIME / f'{len(records)}.files.nul'
        manifest.write_bytes(b'\0'.join(files) + b'\0')
        run(SSH + ['mkdir -p -- ' + shlex.quote(str(root))])
        # Uninitialized submodules still need their empty working directories;
        # otherwise Git reports deletions although their index entries exist.
        if gitlinks:
            run(SSH + ['mkdir -p -- ' + shlex.join(map(str, gitlinks))])
        run(RSYNC + ['--from0', '--files-from=' + str(manifest), str(root) + '/',
                     'root@172.31.224.142:' + str(root) + '/'])
        records.append({**fields, 'tracked_files': len(files)})
    if not args.skip_thirdparty:
        run(RSYNC + ['--exclude=__pycache__', '--exclude=*.pyc',
                     str(ROOT / 'thirdparty') + '/',
                     'root@172.31.224.142:/mnt/d/python/thirdparty/'])
    # This shared repository also contains branch and worktree registrations.
    # No Git writes should run concurrently with this final metadata copy.
    run(RSYNC + ['--exclude=*.lock', str(ROOT / '.git') + '/',
                 'root@172.31.224.142:/mnt/d/python/.git/'])
    report = {'finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'worktrees': records, 'thirdparty_copied': not args.skip_thirdparty}
    (RUNTIME / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
