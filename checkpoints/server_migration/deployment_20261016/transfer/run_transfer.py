#!/usr/bin/env python3
"""Run the explicit migration queue, one rsync at a time, with resumable state.

Source data is never modified. State and stdout are kept outside the repository.
Stop with SIGTERM; rerun to skip completed assets and resume unfinished ones.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import time


BASE = Path(__file__).resolve().parent
RUNTIME = Path('/mnt/d/migration-transfer-logs')
STATE = RUNTIME / 'status.json'
PROGRESS = re.compile(r'^\s*([\d,]+)\s+(\d+)%\s+(\S+)\s+(\S+)')
EXCLUDES = [
    '.git', '__pycache__', '*.pyc', 'browser_profile', 'browser_profiles',
    '*browser-profile*', '.ssh', '.aws', '.azure', '.config/gcloud',
    '.netrc', '.npmrc', '.pypirc', '.env', '.env.*', 'auth.json',
    'credentials.json', 'credentials', 'token', 'stored_tokens',
    'SingletonCookie', 'SingletonSocket', 'SingletonLock',
]
STOP = False
ACTIVE = None


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stop(signum, frame):
    global STOP
    STOP = True
    if ACTIVE is not None and ACTIVE.poll() is None:
        ACTIVE.terminate()


def save(state):
    state['updated_at_utc'] = now()
    temp = STATE.with_suffix('.json.tmp')
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
    temp.replace(STATE)


def command(item, manifest):
    ssh = ['ssh', '-S', manifest['control_path'], '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30',
           '-o', 'ServerAliveCountMax=6', '-p', str(manifest['ssh_port'])]
    cmd = ['rsync', '-aR', '--protect-args', '--partial',
           '--partial-dir=.rsync-partial', '--timeout=300',
           '--info=progress2', '--outbuf=L', '--stats', '-e', shlex.join(ssh)]
    for pattern in EXCLUDES + item.get('exclude', []):
        cmd += ['--exclude', pattern]
    if item.get('include_features_only'):
        cmd += ['--include=*/', '--include=features.npy', '--exclude=*']
    if item.get('compression') == 'zstd-1':
        cmd += ['--compress-choice=zstd', '--compress-level=1']
    cmd += item['paths'] + [manifest['destination']]
    return cmd


def transfer(item, manifest, state):
    global ACTIVE
    entry = state['assets'].setdefault(item['id'], {})
    attempt = entry.get('attempt', 0) + 1
    log = RUNTIME / f"{item['id']}.attempt-{attempt}.log"
    entry.update(status='running', attempt=attempt, started_at_utc=now(),
                 paths=item['paths'], phase=item['phase'], log=str(log),
                 progress=None)
    state['current_asset'] = item['id']
    cmd = command(item, manifest)
    entry['command'] = cmd
    started = time.monotonic()
    with log.open('ab', buffering=0) as stream:
        stream.write((shlex.join(cmd) + '\n').encode())
        ACTIVE = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        entry['rsync_pid'] = ACTIVE.pid
        save(state)
        selector = selectors.DefaultSelector()
        selector.register(ACTIVE.stdout, selectors.EVENT_READ)
        pending = ''
        last_save = started
        while selector.get_map():
            for key, _ in selector.select(timeout=1):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                stream.write(chunk)
                pending += chunk.decode('utf-8', errors='replace')
                records = re.split(r'[\r\n]', pending)
                pending = records.pop()
                for record in records:
                    match = PROGRESS.match(record)
                    if match:
                        entry['progress'] = {
                            'transferred_bytes': int(match[1].replace(',', '')),
                            'percent_of_current_file_list': int(match[2]),
                            'rsync_speed': match[3], 'rsync_time': match[4],
                            'observed_at_utc': now(),
                        }
            if time.monotonic() - last_save >= 5:
                entry['elapsed_seconds'] = round(time.monotonic() - started, 1)
                save(state)
                last_save = time.monotonic()
        code = ACTIVE.wait()
    ACTIVE = None
    entry.update(exit_code=code, finished_at_utc=now(),
                 elapsed_seconds=round(time.monotonic() - started, 1),
                 status='completed' if code == 0 else ('interrupted' if STOP else 'failed'))
    # Completed means rsync succeeded; it does not imply a separate full hash audit.
    save(state)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show-command', metavar='ASSET_ID')
    args = parser.parse_args()
    manifest = json.loads((BASE / 'manifest.json').read_text())
    if args.show_command:
        item = next(q for q in manifest['queue'] if q['id'] == args.show_command)
        print(shlex.join(command(item, manifest)))
        return
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'queue.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another transfer queue is already running')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        state = json.loads(STATE.read_text()) if STATE.exists() else {'assets': {}}
        state.update(pid=os.getpid(), status='running', started_at_utc=now(),
                     manifest=str(BASE / 'manifest.json'), runner_version='20261016-zstd1')
        save(state)
        for item in manifest['queue']:
            if STOP:
                break
            # Reload each group's explicit settings at its boundary, allowing
            # documented policy updates without interrupting an active rsync.
            latest = json.loads((BASE / 'manifest.json').read_text())
            item = next(q for q in latest['queue'] if q['id'] == item['id'])
            if state['assets'].get(item['id'], {}).get('status') == 'completed':
                continue
            # Retry transient transport failures indefinitely: authentication is
            # supplied by the user's existing SSH master, never stored here.
            # Persistent file errors get three attempts, remain failed, then the
            # queue proceeds to independent assets instead of stalling all work.
            failures = 0
            while not STOP:
                code = transfer(item, manifest, state)
                if code == 0 or STOP:
                    break
                failures += 1
                transport = code in {10, 12, 30, 35, 255}
                if failures >= 3 and not transport:
                    break
                state['assets'][item['id']]['status'] = 'retry_wait'
                save(state)
                for _ in range(30):
                    if STOP:
                        break
                    time.sleep(1)
        all_done = all(state['assets'].get(q['id'], {}).get('status') == 'completed'
                       for q in manifest['queue'])
        state.update(status='interrupted' if STOP else ('completed' if all_done else 'needs_attention'),
                     current_asset=None, finished_at_utc=now())
        save(state)


if __name__ == '__main__':
    main()
