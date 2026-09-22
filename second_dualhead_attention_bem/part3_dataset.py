from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class SingleGraphDirectoryDataset(Dataset):
    """
    从指定目录中读取图数据的数据集类。

    目录结构示例：
        dataset_rect_bem_gnn/
            train/
                sample_000.pt
            val/
                sample_000.pt
            test/
                sample_000.pt

    每个 .pt 文件中应保存一个 torch_geometric.data.Data 对象。
    """

    def __init__(self, root_dir: str, split: str):
        """
        初始化数据集。

        参数:
            root_dir: 数据集根目录
            split: 数据划分名称，可选 "train"、"val"、"test"
        """
        self.root_dir = Path(root_dir)
        self.split = split

        # 获取当前划分目录下所有 .pt 文件，并按名称排序
        self.files: List[Path] = sorted((self.root_dir / split).glob("*.pt"))

        # 若没有找到文件，则报错
        if not self.files:
            raise FileNotFoundError(
                f"No .pt files found in {(self.root_dir / split).resolve()}"
            )

    def __len__(self) -> int:
        """
        返回数据集中的样本数量。
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        """
        根据索引读取一个图样本。

        参数:
            idx: 样本索引

        返回:
            data: 一个 torch_geometric.data.Data 对象
        """
        # -----------------------------
        # 重要说明（PyTorch 2.6 兼容）：
        # -----------------------------
        # PyTorch 2.6 开始，torch.load 默认参数 weights_only=True
        # 这会导致“只能安全加载权重类对象”，而不能直接加载完整的 PyG Data 对象。
        #
        # 你的 .pt 文件是自己生成的，保存的是完整图对象：
        #     torch.save(graph, "sample_000.pt")
        #
        # 因此这里必须显式设置：
        #     weights_only=False
        #
        # 否则会报错：
        #     _pickle.UnpicklingError: Weights only load failed
        data = torch.load(
            self.files[idx],
            map_location="cpu",
            weights_only=False
        )

        # 检查读取到的对象类型是否正确
        if not isinstance(data, Data):
            raise TypeError(
                f"{self.files[idx]} does not contain a torch_geometric.data.Data object"
            )

        return data


def build_loaders(root_dir: str = "dataset_rect_bem_gnn", batch_size: int = 1):
    """
    构建训练、验证、测试三个数据加载器。

    参数:
        root_dir: 数据集根目录
        batch_size: 批大小，默认 1

    返回:
        train_loader, val_loader, test_loader
    """
    # 分别构建 train / val / test 数据集
    train_set = SingleGraphDirectoryDataset(root_dir, "train")
    val_set = SingleGraphDirectoryDataset(root_dir, "val")
    test_set = SingleGraphDirectoryDataset(root_dir, "test")

    # 由于每个图都携带各自独立的 H/G 稠密矩阵，
    # 对于当前单图训练场景，batch_size=1 最稳妥
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """
    直接运行本文件时，做一个简单的数据读取测试。
    """
    tr, va, te = build_loaders()

    print("train batches:", len(tr))
    print("val batches:", len(va))
    print("test batches:", len(te))

    # 读取训练集中的第一个 batch
    first = next(iter(tr))

    print("x shape:", tuple(first.x.shape))                    # 节点特征矩阵形状
    print("edge_index shape:", tuple(first.edge_index.shape))  # 边索引矩阵形状

    # 如果你的 Data 里有 edge_weight，也可以顺便查看
    if hasattr(first, "edge_weight"):
        print("edge_weight shape:", tuple(first.edge_weight.shape))

    # 查看几个关键字段是否存在
    print("Has H_mat:", hasattr(first, "H_mat"))
    print("Has G_mat:", hasattr(first, "G_mat"))
    print("Has T_true:", hasattr(first, "T_true"))
    print("Has q_true:", hasattr(first, "q_true"))
