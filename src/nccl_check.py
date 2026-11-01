"""Cek bahwa DDP/NCCL berjalan di semua GPU. Jalankan: torchrun --nproc_per_node=2 src/nccl_check.py"""
import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
x = torch.ones(1024, 1024, device="cuda") * (rank + 1)
dist.all_reduce(x)
expected = sum(range(1, dist.get_world_size() + 1))
print(f"rank {rank} ({torch.cuda.get_device_name(rank)}): {x[0, 0].item()} (harus {expected})")
dist.destroy_process_group()
