#!/usr/bin/env bash
set -euo pipefail
/mnt/d/anaconda3/bin/python - <<'PY'
from pathlib import Path
from ruamel.yaml import YAML
yaml = YAML()
for path in (Path('/root/.condarc'), Path('/mnt/d/anaconda3/.condarc')):
    config = yaml.load(path.read_text()) if path.exists() else {}
    config = config or {}
    config['channels'] = ['conda-forge']
    config['channel_priority'] = 'strict'
    config['auto_activate'] = False
    with path.open('w') as stream:
        yaml.dump(config, stream)
PY
if [[ ! -e /usr/local/bin/conda ]]; then
  ln -s /mnt/d/anaconda3/condabin/conda /usr/local/bin/conda
fi
/mnt/d/anaconda3/bin/conda init bash zsh
