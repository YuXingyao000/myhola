"""Fresh-process import order and image/CAD I/O checks for the target envs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


CHILD = r'''
import importlib, json, pathlib, sys, tempfile
for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
import torch, torchvision
from torchvision.io import encode_jpeg, decode_jpeg
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.IGESControl import IGESControl_Writer, IGESControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs

pixels = torch.full((3, 16, 16), 128, dtype=torch.uint8)
encoded = encode_jpeg(pixels, quality=95)
decoded = decode_jpeg(encoded)
assert torch.equal(pixels, decoded)
box = BRepPrimAPI_MakeBox(1, 2, 3).Shape()
assert BRepCheck_Analyzer(box).IsValid()
with tempfile.TemporaryDirectory() as temp:
    path = str(pathlib.Path(temp) / 'box.igs')
    writer = IGESControl_Writer()
    assert writer.AddShape(box)
    assert writer.Write(path)
    reader = IGESControl_Reader()
    assert reader.ReadFile(path) == IFSelect_RetDone
    assert reader.TransferRoots() > 0
    assert not reader.OneShape().IsNull()
    path = str(pathlib.Path(temp) / 'box.step')
    writer = STEPControl_Writer()
    assert writer.Transfer(box, STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(path) == IFSelect_RetDone
    reader = STEPControl_Reader()
    assert reader.ReadFile(path) == IFSelect_RetDone
    assert reader.TransferRoots() > 0
    assert BRepCheck_Analyzer(reader.OneShape()).IsValid()
libraries = sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines()
                    if 'libjpeg' in line or 'libtiff' in line})
assert str(pathlib.Path(sys.prefix) / 'lib/libjpeg.so.8.3.2') in libraries
assert any('/torchvision.libs/libjpeg.' in p for p in libraries)
print('RESULT_JSON=' + json.dumps(dict(status='passed', prefix=sys.prefix,
    jpeg_roundtrip_exact=True, iges_roundtrip=True, step_roundtrip_valid=True, libraries=libraries)))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    os.chdir('/mnt/d/python')
    variants = {
        'torchvision_first': ['torchvision', 'OCC.Core.IGESControl'],
        'occ_first': ['OCC.Core.IGESControl', 'torchvision'],
        'project_training_then_post': ['src.brepnet.train', 'src.brepnet.dataset',
            'src.brepnet.models.vae', 'src.brepnet.post.construct_brep',
            'src.brepnet.post.occ_fit', 'src.brepnet.post.occ_topology',
            'src.brepnet.eval.run', 'src.brepnet.data.extract_brep'],
        'flux_then_occ': ['diffusers', 'transformers', 'torchvision', 'open3d', 'OCC.Core.IGESControl'],
    }
    report = dict(python=sys.executable, variants={})
    for name, modules in variants.items():
        proc = subprocess.run([sys.executable, '-c', CHILD, json.dumps(modules)],
                              text=True, capture_output=True, timeout=240)
        results = [line.removeprefix('RESULT_JSON=') for line in proc.stdout.splitlines()
                   if line.startswith('RESULT_JSON=')]
        entry = dict(modules=modules, returncode=proc.returncode, stderr=proc.stderr)
        if results:
            entry.update(json.loads(results[-1]))
        else:
            entry.update(status='failed', stdout=proc.stdout)
        report['variants'][name] = entry
        print(name, entry['status'], flush=True)
    report['status'] = 'passed' if all(v['returncode'] == 0 and v['status'] == 'passed'
                                      for v in report['variants'].values()) else 'failed'
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    assert report['status'] == 'passed', report


if __name__ == '__main__':
    main()
