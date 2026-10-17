"""A single NCCL all-reduce on all visible GPUs; no model or dataset access."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(seconds=90),device_id=torch.device('cuda',rank))
x = torch.tensor([float(rank+1)],device=f'cuda:{rank}')
dist.all_reduce(x)
assert x.item() == dist.get_world_size() * (dist.get_world_size()+1) / 2
print(f'GPU {rank}: NCCL all_reduce={x.item()}',flush=True)
dist.destroy_process_group()
