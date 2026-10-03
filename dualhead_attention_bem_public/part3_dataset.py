"""Dataset and PyTorch Geometric loader utilities for the reconstruction experiment."""

from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class SingleGraphDirectoryDataset(Dataset):

    def __init__(self, root_dir: str, split: str):
        self.root_dir = Path(root_dir)
        self.split = split
        self.files: List[Path] = sorted((self.root_dir / split).glob('*.pt'))
        if not self.files:
            raise FileNotFoundError(f'No .pt files found in {(self.root_dir / split).resolve()}')

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        if not isinstance(data, Data):
            raise TypeError(f'{self.files[idx]} does not contain a torch_geometric.data.Data object')
        return data

def build_loaders(root_dir: str='dataset_rect_bem_gnn', batch_size: int=1):
    train_set = SingleGraphDirectoryDataset(root_dir, 'train')
    val_set = SingleGraphDirectoryDataset(root_dir, 'val')
    test_set = SingleGraphDirectoryDataset(root_dir, 'test')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return (train_loader, val_loader, test_loader)
if __name__ == '__main__':
    'Runtime message'
    tr, va, te = build_loaders()
    print('train batches:', len(tr))
    print('val batches:', len(va))
    print('test batches:', len(te))
    first = next(iter(tr))
    print('x shape:', tuple(first.x.shape))
    print('edge_index shape:', tuple(first.edge_index.shape))
    if hasattr(first, 'edge_weight'):
        print('edge_weight shape:', tuple(first.edge_weight.shape))
    print('Has H_mat:', hasattr(first, 'H_mat'))
    print('Has G_mat:', hasattr(first, 'G_mat'))
    print('Has T_true:', hasattr(first, 'T_true'))
    print('Has q_true:', hasattr(first, 'q_true'))
