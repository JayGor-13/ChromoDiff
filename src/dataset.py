import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GenomicDataset(Dataset):
    """
    Wraps a tensor of genomic sequences represented as token indices.
    """
    def __init__(self, data: torch.Tensor):
        if not isinstance(data, torch.Tensor):
            self.data = torch.tensor(data, dtype=torch.long)
        else:
            self.data = data.long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def reverse_complement_tokens(x: np.ndarray) -> np.ndarray:
    """
    Reverse complement sequence array in NumPy.
    Mapping: A(0)->T(3), C(1)->G(2), G(2)->C(1), T(3)->A(0), N(4)->N(4), [MASK](5)->[MASK](5)
    """
    comp_map = np.array([3, 2, 1, 0, 4, 5], dtype=np.int8)
    return comp_map[x[:, ::-1]]

def reverse_complement_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Reverse complement sequence tensor in PyTorch.
    """
    comp_map = torch.tensor([3, 2, 1, 0, 4, 5], dtype=torch.long, device=x.device)
    reversed_x = torch.flip(x, dims=[-1])
    return comp_map[reversed_x]

import os
from torch.utils.data.distributed import DistributedSampler

def get_dataloader(data_path: str, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Load sequence token array from disk, wrap it in a GenomicDataset, and return a DataLoader.
    """
    data_np = np.load(data_path)
    data_tensor = torch.tensor(data_np, dtype=torch.long)
    dataset = GenomicDataset(data_tensor)
    
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True if sampler is None and shuffle else False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers
    )
    return loader
