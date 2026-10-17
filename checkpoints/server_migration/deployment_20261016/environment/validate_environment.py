"""Dependency and hardware checks; does not read research datasets or train models."""
from __future__ import annotations
import importlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import torch
from torch_scatter import scatter_mean
from torch_sparse import SparseTensor
from torch_geometric.nn import GATv2Conv

result = {
    'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
    'architectures': torch.cuda.get_arch_list(), 'devices': [],
}
assert torch.cuda.is_available()
assert 'sm_120' in torch.cuda.get_arch_list(), result['architectures']
for device_id in range(torch.cuda.device_count()):
    device = torch.device('cuda', device_id)
    torch.manual_seed(17)
    x = torch.randn(64, 16, device=device, requires_grad=True)
    weights = torch.randn(16, 32, device=device, requires_grad=True)
    loss = (x @ weights).square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(weights.grad).all()
    bf16 = (x.detach().bfloat16() @ weights.detach().bfloat16()).float()
    assert torch.isfinite(bf16).all()
    index = torch.arange(64, device=device) % 4
    reduced = scatter_mean(x.detach(), index, dim=0)
    expected = torch.stack([x.detach()[index == i].mean(0) for i in range(4)])
    torch.testing.assert_close(reduced, expected)
    row = torch.tensor([0, 1, 2], device=device)
    col = torch.tensor([1, 2, 0], device=device)
    sparse = SparseTensor(row=row, col=col, value=torch.ones(3, device=device), sparse_sizes=(3,3))
    dense = torch.randn(3, 5, device=device)
    torch.testing.assert_close(sparse.matmul(dense), dense[[1,2,0]])
    graph = GATv2Conv(16, 8, heads=2).to(device)
    edge_index = torch.stack((torch.arange(64, device=device), torch.roll(torch.arange(64,device=device),1)))
    graph_x = torch.randn(64,16,device=device,requires_grad=True)
    graph(graph_x,edge_index).square().mean().backward()
    assert torch.isfinite(graph_x.grad).all()
    torch.cuda.synchronize(device)
    result['devices'].append({'index':device_id,'name':torch.cuda.get_device_name(device),
       'capability':torch.cuda.get_device_capability(device),'matmul_backward':True,
       'bf16':True,'scatter_mean':True,'sparse_matmul':True,'gatv2_backward':True})

from chamferdist import ChamferDistance
cloud_x = torch.tensor([[[0.,0.,0.],[1.,0.,0.]]], device='cuda:0', requires_grad=True)
cloud_y = torch.tensor([[[0.,0.,0.],[2.,0.,0.]]], device='cuda:0')
cd = ChamferDistance()(cloud_x, cloud_y, bidirectional=True, point_reduction='mean', batch_reduction='mean')
torch.testing.assert_close(cd, torch.tensor(1., device='cuda:0'))
cd.backward()
assert torch.isfinite(cloud_x.grad).all()
result['chamferdist_cuda'] = {'distance': cd.item(), 'backward': True}

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
import OCC
assert OCC.VERSION.startswith('7.8.1'), OCC.VERSION
shape = BRepPrimAPI_MakeBox(1,2,3).Shape()
assert BRepCheck_Analyzer(shape).IsValid()
with tempfile.TemporaryDirectory() as tmp:
    path = str(Path(tmp) / 'box.step')
    writer = STEPControl_Writer()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(path) == IFSelect_RetDone
    reader = STEPControl_Reader()
    assert reader.ReadFile(path) == IFSelect_RetDone
    assert reader.TransferRoots() > 0
    assert BRepCheck_Analyzer(reader.OneShape()).IsValid()
result['occ'] = {'version': OCC.VERSION, 'box_step_roundtrip_valid': True}

from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from transformers import T5EncoderModel, CLIPTextModel, Dinov2Model
import gguf, sentencepiece, hydra, pytorch_lightning, open3d, ray
GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
result['flux_imports'] = True
import numpy as np
from gguf.quants import quantize
from gguf import GGMLQuantizationType
from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
q8_source = np.linspace(-1, 1, 16*64, dtype=np.float32).reshape(16,64)
q8_packed = quantize(q8_source, GGMLQuantizationType.Q8_0)
q8_weight = GGUFParameter(torch.from_numpy(q8_packed).to('cuda:0'), quant_type=GGMLQuantizationType.Q8_0)
q8_restored = dequantize_gguf_tensor(q8_weight).float()
q8_expected = torch.from_numpy(q8_source).to('cuda:0')
torch.testing.assert_close(q8_restored, q8_expected, atol=0.01, rtol=0)
result['gguf_q8_cuda'] = {'shape': list(q8_restored.shape), 'max_abs_error': (q8_restored-q8_expected).abs().max().item()}
result['versions'] = {name: importlib.metadata.version(name) for name in (
    'torch','torchvision','torch-geometric','torch-scatter','torch-sparse','pytorch-lightning',
    'numpy','chamferdist','pyg-lib','diffusers','transformers','gguf','sentencepiece','hydra-core','accelerate','open3d')}
Path('/mnt/d/migration/logs/environment-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
