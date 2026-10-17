#!/usr/bin/env python3
"""Restricted SSH endpoint and conservative Git receive hooks for one checkout."""
import os
from pathlib import Path
import subprocess
import sys


def git(repo, *args, check=True):
    result = subprocess.run(['git', f'--git-dir={repo / ".git"}', f'--work-tree={repo}', *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace').strip() or 'Git command failed')
    return result


def reject(reason):
    raise RuntimeError('hola-sync: ' + reason)


def main(mode, repo):
    repo = Path(repo).resolve()
    if mode == 'serve':
        allowed = {f"git-receive-pack '{repo}'", f'git-receive-pack {repo}'}
        if os.environ.get('SSH_ORIGINAL_COMMAND') not in allowed:
            reject('This key can only push to the configured project')
        os.execv('/usr/bin/git-receive-pack', ['git-receive-pack', str(repo)])
    if mode == 'pre-receive':
        updates = [line.split() for line in sys.stdin.read().splitlines()]
        current = git(repo, 'symbolic-ref', '-q', 'HEAD').stdout.decode().strip()
        if len(updates) != 1 or len(updates[0]) != 3:
            reject('Only one branch update is allowed')
        old, new, branch = updates[0]
        if branch != current or old == '0' * 40 or new == '0' * 40:
            reject('Target must already be on the same branch; no branch creation/deletion')
        if git(repo, 'merge-base', '--is-ancestor', old, new, check=False).returncode:
            reject('Non-fast-forward update; reconcile histories manually')
        return
    if mode != 'push-to-checkout':
        reject('Unknown mode')
    head = sys.argv[3]
    for marker in ['MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'rebase-merge', 'rebase-apply', 'sequencer']:
        if (repo / '.git' / marker).exists():
            reject('Target has a Git operation in progress')
    git(repo, 'update-index', '-q', '--refresh')
    if git(repo, 'status', '--porcelain', '--untracked-files=normal').stdout:
        reject('Target has local modifications or untracked files; preserving them')
    # Even ignored data must not be overwritten by a newly tracked path.
    added = git(repo, 'diff', '--name-only', '-z', '--no-renames', '--diff-filter=A', 'HEAD', head, '--').stdout
    for raw in added.split(b'\0'):
        if not raw:
            continue
        path = repo / os.fsdecode(raw)
        if os.path.lexists(path):
            reject(f'Incoming tracked path collides with existing data: {path}')
        for parent in path.parents:
            if parent == repo:
                break
            if parent.is_symlink():
                reject(f'Incoming path traverses an existing symlink: {parent}')
    # This is Git's documented two-tree checkout; it preserves unrelated ignored data.
    git(repo, 'read-tree', '-u', '-m', 'HEAD', head)


if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
