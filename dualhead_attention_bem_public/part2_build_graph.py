"""Build the fully connected boundary graph and node features used by the model."""

from pathlib import Path
import torch
from torch_geometric.data import Data
from part1_bem_solver import build_problem_definition

def build_full_graph(centers: torch.Tensor):
    num_nodes = centers.size(0)
    src = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes, dtype=torch.long).repeat(num_nodes)
    edge_index = torch.stack([src, dst], dim=0)
    diff = centers[src] - centers[dst]
    dist = torch.norm(diff, dim=1)
    edge_weight = 1.0 / (1.0 + dist)
    return (edge_index, edge_weight)

def build_single_graph_data() -> Data:
    raw = build_problem_definition(n_bottom=20, n_right=11, n_top=20, n_left=11)
    centers = torch.tensor(raw['centers'], dtype=torch.float32)
    normals = torch.tensor(raw['normals'], dtype=torch.float32)
    lengths = torch.tensor(raw['lengths'], dtype=torch.float32)
    H = torch.tensor(raw['H'], dtype=torch.float32)
    G = torch.tensor(raw['G'], dtype=torch.float32)
    bc_type = torch.tensor(raw['bc_type'], dtype=torch.long)
    bc_value = torch.tensor(raw['bc_value'], dtype=torch.float32)
    T_true = torch.tensor(raw['T_true'], dtype=torch.float32)
    q_true = torch.tensor(raw['q_true'], dtype=torch.float32)
    boundary_nodes = torch.tensor(raw['boundary_nodes'], dtype=torch.float32)
    mask_T_known = (bc_type == 0).float()
    mask_q_known = (bc_type == 1).float()
    T_known = torch.zeros_like(T_true)
    q_known = torch.zeros_like(q_true)
    T_known[mask_T_known > 0.5] = bc_value[mask_T_known > 0.5]
    q_known[mask_q_known > 0.5] = bc_value[mask_q_known > 0.5]
    known_T_feature = T_known.unsqueeze(1)
    known_q_feature = q_known.unsqueeze(1)
    x = torch.cat([centers, normals, lengths.unsqueeze(1), mask_T_known.unsqueeze(1), mask_q_known.unsqueeze(1), known_T_feature, known_q_feature], dim=1)
    edge_index, edge_weight = build_full_graph(centers)
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, centers=centers, normals=normals, lengths=lengths, H_mat=H, G_mat=G, bc_type=bc_type, bc_value=bc_value, T_known=T_known, q_known=q_known, mask_T_known=mask_T_known, mask_q_known=mask_q_known, T_true=T_true, q_true=q_true, boundary_nodes=boundary_nodes)
    return data

def save_dataset(root_dir: str='dataset_rect_bem_gnn'):
    root = Path(root_dir)
    for split in ['train', 'val', 'test']:
        (root / split).mkdir(parents=True, exist_ok=True)
    graph = build_single_graph_data()
    torch.save(graph, root / 'train' / 'sample_000.pt')
    torch.save(graph, root / 'val' / 'sample_000.pt')
    torch.save(graph, root / 'test' / 'sample_000.pt')
    print(f'Saved dataset to: {root.resolve()}')
    print('train/val/test each contains exactly 1 graph sample.')
if __name__ == '__main__':
    save_dataset()
