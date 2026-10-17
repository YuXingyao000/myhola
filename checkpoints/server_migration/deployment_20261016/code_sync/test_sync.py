"""Real disposable Git repositories: checkout protection and async hook behavior."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hola-sync-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.target = self.root / 'source', self.root / 'target'
        self.git(self.root, 'init', '-b', 'main', str(self.source))
        self.git(self.source, 'config', 'user.name', 'Sync test')
        self.git(self.source, 'config', 'user.email', 'sync@example.invalid')
        (self.source / '.gitignore').write_text('cache/\n')
        self.commit('initial')
        self.git(self.root, 'clone', str(self.source), str(self.target))
        for key, value in [('receive.denyCurrentBranch', 'updateInstead'),
                           ('receive.denyNonFastForwards', 'true'), ('receive.denyDeletes', 'true'),
                           ('user.name', 'Sync test'), ('user.email', 'sync@example.invalid')]:
            self.git(self.target, 'config', key, value)
        for mode in ['pre-receive', 'push-to-checkout']:
            hook = self.target / '.git/hooks' / mode
            hook.write_text('#!/bin/sh\nexec ' + shlex.join([
                sys.executable, str(HERE / 'receiver.py'), mode, str(self.target)]) + ' "$@"\n')
            hook.chmod(0o755)

    def git(self, repo, *args, check=True):
        return subprocess.run(['git', '-C', str(repo), *args], check=check, capture_output=True, text=True)

    def commit(self, content):
        (self.source / 'code.txt').write_text(content)
        self.git(self.source, 'add', '.')
        self.git(self.source, 'commit', '-m', content)

    def push(self):
        return self.git(self.source, 'push', str(self.target), 'HEAD:refs/heads/main', check=False)

    def test_fast_forward_and_ignored_data(self):
        (self.target / 'cache').mkdir()
        (self.target / 'cache/data').write_text('preserve')
        self.commit('next')
        self.assertEqual(self.push().returncode, 0)
        self.assertEqual((self.target / 'code.txt').read_text(), 'next')
        self.assertEqual((self.target / 'cache/data').read_text(), 'preserve')
        self.assertEqual(self.git(self.target, 'status', '--porcelain').stdout, '')

    def test_dirty_target(self):
        (self.target / 'code.txt').write_text('local work')
        self.commit('next')
        self.assertNotEqual(self.push().returncode, 0)
        self.assertEqual((self.target / 'code.txt').read_text(), 'local work')

    def test_untracked_target(self):
        (self.target / 'local.txt').write_text('local work')
        self.commit('next')
        self.assertNotEqual(self.push().returncode, 0)
        self.assertEqual((self.target / 'local.txt').read_text(), 'local work')

    def test_ignored_collision(self):
        (self.target / 'cache').mkdir()
        (self.target / 'cache/data').write_text('local data')
        (self.source / 'cache').mkdir()
        (self.source / 'cache/data').write_text('incoming')
        self.git(self.source, 'add', '-f', 'cache/data')
        self.commit('next')
        self.assertNotEqual(self.push().returncode, 0)
        self.assertEqual((self.target / 'cache/data').read_text(), 'local data')

    def test_different_branch(self):
        self.git(self.target, 'checkout', '-b', 'other')
        self.commit('next')
        self.assertNotEqual(self.push().returncode, 0)
        self.assertEqual((self.target / 'code.txt').read_text(), 'initial')

    def test_diverged_target(self):
        (self.target / 'code.txt').write_text('remote commit')
        self.git(self.target, 'commit', '-am', 'remote commit')
        self.commit('next')
        self.assertNotEqual(self.push().returncode, 0)
        self.assertEqual((self.target / 'code.txt').read_text(), 'remote commit')

    def test_restricted_key(self):
        result = subprocess.run([sys.executable, str(HERE / 'receiver.py'), 'serve', str(self.target)],
                                env={**os.environ, 'SSH_ORIGINAL_COMMAND': 'id'}, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_async_commit_hook_and_worktree_scope(self):
        folder = self.source / '.git/hola-sync'
        folder.mkdir()
        (folder / 'config.json').write_text(json.dumps({
            'enabled': True, 'source': str(self.source), 'target': str(self.target), 'identity': '/unused'}))
        hook = self.source / '.git/hooks/post-commit'
        hook.write_text('#!/bin/sh\n' + shlex.join([sys.executable, str(HERE / 'sender.py'), 'enqueue']) +
                        ' "$(git rev-parse --show-toplevel)"\nexit 0\n')
        hook.chmod(0o755)
        for index in range(3):
            self.commit(f'rapid {index}')
        expected = self.git(self.source, 'rev-parse', 'HEAD').stdout.strip()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if (folder / 'status.json').exists():
                status = json.loads((folder / 'status.json').read_text())
                if status.get('status') == 'synced' and status.get('commit') == expected:
                    break
            time.sleep(.1)
        else:
            self.fail((folder / 'worker.log').read_text())
        self.assertEqual(self.git(self.target, 'rev-parse', 'HEAD').stdout.strip(), expected)
        linked = self.root / 'linked'
        self.git(self.source, 'worktree', 'add', '-b', 'linked', str(linked))
        (linked / 'code.txt').write_text('other worktree')
        self.git(linked, 'commit', '-am', 'other worktree')
        self.assertEqual(json.loads((folder / 'status.json').read_text())['commit'], expected)
        # Wait for queued workers to release their log handles before cleanup.
        time.sleep(.3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
