from torch.utils.data import Dataset
from bisect import bisect
from typing import Tuple
from pathlib import Path
import numpy as np
import torch

class TokenPrefixDataset(Dataset):
    def __init__(self, data_path: str, normalize_prefix: bool = False):
        super().__init__()
        
        self.normalize_prefix = normalize_prefix
        
        path = Path(data_path)
        images_path = path / "img_embeddings"
        tokens_path = path / "text_tokens"
        masks_path = path / "text_masks"
        
        embedding_files = sorted(list(images_path.glob("*.npy")), key=lambda x: x.name)
        token_files = sorted(list(tokens_path.glob("*.npy")), key=lambda x: x.name)
        mask_files = sorted(list(masks_path.glob("*.npy")), key=lambda x: x.name)
        
        self.embedding_file_data = [np.load(file, mmap_mode='r') for file in embedding_files]
        self.token_file_data = [np.load(file, mmap_mode='r') for file in token_files]
        self.mask_file_data = [np.load(file, mmap_mode='r') for file in mask_files]
        
        self.start_indices = [0] * len(self.embedding_file_data)
        self.sample_count = 0
        
        for i, memmap in enumerate(self.embedding_file_data):
            self.start_indices[i] = self.sample_count
            self.sample_count += memmap.shape[0]
    
    def __len__(self):
        return self.sample_count
    
    def __getitem__(self, sample_index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_index = bisect(self.start_indices, sample_index) - 1
        memmap_index = sample_index - self.start_indices[batch_index]
        
        tokens = self.token_file_data[batch_index][memmap_index]
        mask = self.mask_file_data[batch_index][memmap_index]
        prefix = self.embedding_file_data[batch_index][memmap_index]
        
        tokens = torch.from_numpy(
            np.array(tokens, dtype=np.int64)
        )
        
        mask = torch.from_numpy(
            np.array(mask, dtype=np.float32)
        )
        
        prefix = torch.from_numpy(
            np.array(prefix, dtype=np.float32)
        )
    
        if self.normalize_prefix:
            prefix = prefix / prefix.norm(2, -1)

        return tokens, mask, prefix