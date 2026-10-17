#!/usr/bin/env python3
"""Project-local, asynchronous, append-only LAN Git synchronization."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def clean_env():
    env = os.environ.copy()
    # A Git hook may export index/worktree variables; the worker uses an explicit repo.
    for name in subprocess.check_output(['git', 'rev-parse', '--local-env-vars'], text=True).split():
        env.pop(name, None)
    return env


def run(mode, repo):
    repo = Path(repo).resolve()
    common = Path(git(repo, 'rev-parse', '--path-format=absolute', '--git-common-dir'))
    folder = common / 'hola-sync'
    config_path = folder / 'config.json'
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    # Shared hooks also run in linked worktrees: enable only the explicitly configured root.
    machine = Path('/etc/machine-id').read_text().strip()
    if (str(repo) != config['source'] or not config.get('enabled', False)
            or config.get('source_machine', machine) != machine):
        return
    state = folder / 'status.json'
    if mode == 'status':
        print(state.read_text() if state.exists() else '{"status": "not_started"}')
        return
    if mode == 'enqueue':
        with (folder / 'worker.log').open('ab') as log:
            subprocess.Popen([sys.executable, __file__, 'worker', str(repo)],
                             cwd=repo, env=clean_env(), stdin=subprocess.DEVNULL,
                             stdout=log, stderr=log, start_new_session=True, close_fds=True)
        return
    if mode != 'worker':
        raise ValueError('Expected enqueue, worker, or status')

    def save(status, **extra):
        data = {'status': status, 'updated_at_utc': datetime.now(timezone.utc).isoformat(), **extra}
        tmp = folder / 'status.tmp'
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        tmp.replace(state)
        print(json.dumps(data), flush=True)

    # Parallel commits serialize. Each worker samples the latest HEAD, so queued
    # workers coalesce after a successful push rather than replaying older commits.
    with (folder / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for attempt in range(1, 4):
            config = json.loads(config_path.read_text())
            if not config.get('enabled', False):
                save('paused')
                return
            git_dir = Path(git(repo, 'rev-parse', '--absolute-git-dir'))
            if any((git_dir / item).exists() for item in
                   ['rebase-merge', 'rebase-apply', 'MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'sequencer']):
                save('blocked', reason='Git operation in progress; retry after it finishes')
                return
            try:
                branch = git(repo, 'symbolic-ref', '-q', 'HEAD')
            except subprocess.CalledProcessError:
                save('blocked', reason='Detached HEAD')
                return
            head = git(repo, 'rev-parse', 'HEAD')
            if state.exists():
                previous = json.loads(state.read_text())
                if previous.get('status') == 'synced' and previous.get('commit') == head and previous.get('branch') == branch:
                    return
            save('syncing', commit=head, branch=branch, attempt=attempt)
            env = os.environ.copy()
            env['GIT_TERMINAL_PROMPT'] = '0'
            env['GIT_SSH_COMMAND'] = shlex.join([
                'ssh', '-i', config['identity'], '-o', 'IdentitiesOnly=yes',
                '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'ControlPath=none', '-o', 'ConnectTimeout=10',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
            ])
            try:
                result = subprocess.run(
                    ['git', '-C', str(repo), '-c', 'push.followTags=false',
                     'push', '--porcelain', config['target'], f'{head}:{branch}'],
                    env=env, capture_output=True, text=True, timeout=120)
                detail = (result.stdout + result.stderr)[-8000:]
                if result.returncode == 0:
                    save('synced', commit=head, branch=branch, detail=detail)
                    return
                # Git safety rejections require intervention, not repeated retries.
                rejected = '[rejected]' in detail or '[remote rejected]' in detail
            except subprocess.TimeoutExpired:
                detail, rejected = 'Git push timed out after 120 seconds', False
            if rejected or attempt == 3:
                save('blocked' if rejected else 'failed', commit=head, branch=branch, detail=detail)
                return
            save('retrying', commit=head, branch=branch, attempt=attempt, detail=detail)
            time.sleep(20 if attempt == 1 else 60)


if __name__ == '__main__':
    try:
        run(sys.argv[1], sys.argv[2])
    except Exception as exc:
        print(f'hola-sync: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
