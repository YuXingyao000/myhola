"""Check trusted migrated AE checkpoint deserialization under the new PyTorch."""
import json
from pathlib import Path
import torch
path = Path('/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt')
report = {'path': str(path), 'torch': torch.__version__}
try:
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    report['default_load'] = 'passed'
except Exception as error:
    report['default_load'] = type(error).__name__
    report['default_error'] = str(error)[:1600]
    # This is the user's trusted source AE checkpoint, not an arbitrary upload.
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    report['explicit_weights_only_false'] = 'passed'
report['keys'] = list(checkpoint.keys())[:20]
state = checkpoint.get('state_dict', checkpoint)
report['state_dict_entries'] = len(state)
Path('/mnt/d/migration/logs/checkpoint-compatibility.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
