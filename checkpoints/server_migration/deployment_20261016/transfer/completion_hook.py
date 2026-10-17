#!/usr/bin/env python3
"""Wait locally without model calls, then run one project acceptance agent."""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


RUNTIME = Path('/mnt/d/migration-transfer-logs')
STATE = RUNTIME / 'status.json'
HOOK = RUNTIME / 'acceptance-hook'
PROMPT = Path(__file__).with_name('acceptance_prompt.md')


def trigger(state):
    status = state.get('status')
    if status in {'completed', 'needs_attention'}:
        return status
    # An explicit pause/interruption must not be undone automatically.
    if status == 'interrupted':
        return None
    for entry in state.get('assets', {}).values():
        if entry.get('status') in {'failed', 'retry_wait'} and entry.get('attempt', 0) >= 3:
            return 'repeated_asset_failure'
    pid = state.get('pid')
    if status == 'running' and pid:
        try:
            process_state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            return 'queue_process_exited'
        if process_state == 'Z':
            return 'queue_process_exited'
    return None


def write_status(value):
    value['updated_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    temporary = HOOK / 'status.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(HOOK / 'status.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Inspect trigger; do not launch an agent')
    args = parser.parse_args()
    if args.check:
        print(json.dumps({'trigger': trigger(json.loads(STATE.read_text()))}))
        return
    HOOK.mkdir(parents=True, exist_ok=True)
    with (HOOK / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = HOOK / 'status.json'
        if existing.exists() and json.loads(existing.read_text()).get('phase') in {'launched', 'finished'}:
            return
        write_status({'phase': 'waiting', 'pid': os.getpid()})
        while True:
            try:
                state = json.loads(STATE.read_text())
            except (OSError, json.JSONDecodeError):
                time.sleep(60)
                continue
            reason = trigger(state)
            if reason:
                break
            if state.get('status') == 'interrupted':
                write_status({'phase': 'paused_with_queue', 'pid': os.getpid()})
                return
            time.sleep(60)
        # Mark before launch: a hook restart must not create duplicate paid runs.
        write_status({'phase': 'launched', 'pid': os.getpid(), 'reason': reason})
        env = dict(os.environ)
        env.pop('CODEX_THREAD_ID', None)
        env.pop('CODEX_TURN_ID', None)
        command = ['codex', 'exec', '-C', '/mnt/d/python',
                   '-s', 'danger-full-access', '-c', 'approval_policy="never"',
                   '-c', 'model_reasoning_effort="high"', '--json',
                   '-o', str(HOOK / 'result.md'), '-']
        with PROMPT.open('rb') as prompt, (HOOK / 'agent.jsonl').open('ab') as output, \
                (HOOK / 'agent.stderr').open('ab') as errors:
            try:
                code = subprocess.run(command, stdin=prompt, stdout=output, stderr=errors,
                                      cwd='/mnt/d/python', env=env, check=False).returncode
            except OSError as error:
                write_status({'phase': 'finished', 'reason': reason, 'launch_error': str(error)})
                return
        write_status({'phase': 'finished', 'reason': reason, 'exit_code': code})


if __name__ == '__main__':
    main()
