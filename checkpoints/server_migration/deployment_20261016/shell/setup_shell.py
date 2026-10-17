#!/usr/bin/env python3
"""Install the requested target shell, preserving existing Conda/user config."""

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import urllib.request


ROOT = Path('/root')
REPORT = Path('/mnt/d/migration/shell')
OMZ = ROOT / '.oh-my-zsh'


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'HoLa-server-setup'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def install_fonts():
    selected = [
        ('google/fonts', 'ofl/archivo', 'Archivo', lambda n: n.endswith('.ttf') or n == 'OFL.txt'),
        ('google/fonts', 'ofl/jost', 'Jost', lambda n: n.endswith('.ttf') or n == 'OFL.txt'),
        ('adobe-fonts/source-han-sans', 'OTF/SimplifiedChinese?ref=release', 'SourceHanSans',
         lambda n: n in {'SourceHanSansSC-Regular.otf', 'SourceHanSansSC-Bold.otf'}),
        ('adobe-fonts/source-han-sans', '', 'SourceHanSans', lambda n: n == 'LICENSE.txt'),
    ]
    records = []
    for repo, relative, family, accept in selected:
        entries = json.loads(fetch(f'https://api.github.com/repos/{repo}/contents/{relative}'))
        matching = [entry for entry in entries if accept(entry['name'])]
        if not matching:
            raise RuntimeError(f'No expected font/license files found in {repo}/{relative}')
        folder = Path('/usr/local/share/fonts/hola') / family
        folder.mkdir(parents=True, exist_ok=True)
        for entry in matching:
            destination = folder / entry['name']
            data = destination.read_bytes() if destination.is_file() else b''
            existing_blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if existing_blob != entry['sha']:
                data = fetch(entry['download_url'])
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != entry['sha']:
                raise RuntimeError(f'GitHub blob checksum mismatch: {entry["name"]}')
            destination.write_bytes(data)
            destination.chmod(0o644)
            records.append({'path': str(destination), 'url': entry['download_url'],
                            'git_blob_sha1': blob, 'sha256': hashlib.sha256(data).hexdigest()})
    run('fc-cache', '-f', '/usr/local/share/fonts/hola')
    return records


def main():
    if os.geteuid() != 0:
        raise SystemExit('Run on the target server as root')
    REPORT.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = Path('/mnt/d/migration/backups') / ('shell-' + timestamp)
    backup.mkdir(parents=True)
    original_shell = pwd.getpwnam('root').pw_shell
    for name in ['.zshrc', '.zprofile', '.zshenv', '.p10k.zsh']:
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, backup / name)
    (backup / 'original-shell.txt').write_text(original_shell + '\n')
    repos = [
        ('https://github.com/ohmyzsh/ohmyzsh.git', OMZ),
        ('https://github.com/romkatv/powerlevel10k.git', OMZ / 'custom/themes/powerlevel10k'),
        ('https://github.com/zsh-users/zsh-autosuggestions.git', OMZ / 'custom/plugins/zsh-autosuggestions'),
        ('https://github.com/zsh-users/zsh-syntax-highlighting.git', OMZ / 'custom/plugins/zsh-syntax-highlighting'),
    ]
    repo_records = []
    for url, destination in repos:
        if not destination.exists():
            run('git', 'clone', '--depth=1', url, str(destination))
        actual = subprocess.check_output(['git', '-C', str(destination), 'remote', 'get-url', 'origin'], text=True).strip()
        if actual != url:
            raise RuntimeError(f'Unexpected existing checkout: {destination}')
        commit = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
        repo_records.append({'url': url, 'path': str(destination), 'commit': commit})
    fonts = install_fonts()
    prompt = ROOT / '.p10k.zsh'
    if not prompt.exists():
        prompt.write_text('''# HOLA MIGRATION TEMPORARY PROMPT
# Fonts/colors/layout will be chosen by the user later.
typeset -g POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true
typeset -g POWERLEVEL9K_MODE=ascii
typeset -ga POWERLEVEL9K_LEFT_PROMPT_ELEMENTS=(context dir vcs prompt_char)
typeset -ga POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS=(status command_execution_time anaconda)
typeset -g POWERLEVEL9K_PROMPT_ADD_NEWLINE=false
typeset -g POWERLEVEL9K_PROMPT_CHAR_OK_VIINS_CONTENT_EXPANSION='>'
typeset -g POWERLEVEL9K_PROMPT_CHAR_ERROR_VIINS_CONTENT_EXPANSION='>'
typeset -g POWERLEVEL9K_ANACONDA_SHOW_PYTHON_VERSION=false
typeset -g POWERLEVEL9K_INSTANT_PROMPT=off
''')
    start = '# >>> HoLa interactive shell >>>'
    end = '# <<< HoLa interactive shell <<<'
    block = '''# >>> HoLa interactive shell >>>
[[ -r /etc/profile.d/hola-network.sh ]] && source /etc/profile.d/hola-network.sh
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="powerlevel10k/powerlevel10k"
zstyle ':omz:update' mode disabled
typeset -g POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true
[[ -r "$HOME/.p10k.zsh" ]] && source "$HOME/.p10k.zsh"
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
source "$ZSH/oh-my-zsh.sh"
# <<< HoLa interactive shell <<<'''
    rc = ROOT / '.zshrc'
    existing = rc.read_text() if rc.exists() else ''
    if start in existing:
        before, rest = existing.split(start, 1)
        _, after = rest.split(end, 1)
        updated = before + block + after
    else:
        updated = existing.rstrip() + '\n\n' + block + '\n'
    rc.write_text(updated)
    run('zsh', '-n', str(rc))
    run('zsh', '-n', str(prompt))
    run('zsh', '-lic', '(( $+functions[p10k] && $+functions[_zsh_autosuggest_start] && $+functions[_zsh_highlight] && $+functions[conda] ))')
    run('chsh', '-s', shutil.which('zsh'), 'root')
    report = {'installed_at_utc': timestamp, 'backup': str(backup),
              'original_shell': original_shell, 'current_shell': pwd.getpwnam('root').pw_shell,
              'repositories': repo_records, 'fonts': fonts}
    (REPORT / 'installation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'installed', 'backup': str(backup), 'shell': report['current_shell']}))


if __name__ == '__main__':
    main()
