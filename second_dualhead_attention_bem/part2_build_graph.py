from pathlib import Path
import torch
from torch_geometric.data import Data
from part1_bem_solver import build_problem_definition

def build_full_graph(centers: torch.Tensor):
    """
    根据节点中心坐标构造全连接图（包含自环）。

    参数:
        centers: 形状为 [num_nodes, 2] 的张量，表示每个边界单元中心的坐标

    返回:
        edge_index: 图的边索引，形状为 [2, num_edges]
        edge_weight: 每条边的权重，基于节点间距离计算
    """
    num_nodes = centers.size(0)

    # 构造全连接图的起点和终点索引
    # src: [0,0,0,...,1,1,1,...,num_nodes-1,...]
    # dst: [0,1,2,...,0,1,2,...,0,1,2,...]
    src = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes, dtype=torch.long).repeat(num_nodes)

    # 拼接成 PyG 所需的 edge_index 格式
    edge_index = torch.stack([src, dst], dim=0)

    # 计算每条边两端节点中心的坐标差
    diff = centers[src] - centers[dst]

    # 计算欧氏距离
    dist = torch.norm(diff, dim=1)

    # GCNConv 支持 edge_weight
    # 这里使用 1 / (1 + r) 作为边权重，避免距离很小时数值过大，同时保持有界
    # 当 src == dst 时，自环距离为 0，因此自环权重为 1
    edge_weight = 1.0 / (1.0 + dist)

    return edge_index, edge_weight


def build_single_graph_data() -> Data:
    """
    构建单个图样本的数据对象。
    该图样本来自固定矩形区域的边界元问题定义。
    """
    # 调用边界元求解器中的问题构造函数，生成原始物理问题数据
    raw = build_problem_definition(
        n_bottom=20,  # 底边离散单元数
        n_right=11,    # 右边离散单元数
        n_top=20,     # 顶边离散单元数
        n_left=11,     # 左边离散单元数
    )

    # 将原始 numpy / list 数据转换为 torch 张量
    centers = torch.tensor(raw["centers"], dtype=torch.float32)          # 单元中心坐标
    normals = torch.tensor(raw["normals"], dtype=torch.float32)          # 单元外法向
    lengths = torch.tensor(raw["lengths"], dtype=torch.float32)          # 单元长度
    H = torch.tensor(raw["H"], dtype=torch.float32)                      # BEM H 矩阵
    G = torch.tensor(raw["G"], dtype=torch.float32)                      # BEM G 矩阵
    bc_type = torch.tensor(raw["bc_type"], dtype=torch.long)             # 边界条件类型
    bc_value = torch.tensor(raw["bc_value"], dtype=torch.float32)        # 边界条件数值
    T_true = torch.tensor(raw["T_true"], dtype=torch.float32)            # 真实温度
    q_true = torch.tensor(raw["q_true"], dtype=torch.float32)            # 真实热流
    boundary_nodes = torch.tensor(raw["boundary_nodes"], dtype=torch.float32)  # 边界节点坐标

    # 边界条件掩码：
    # bc_type == 0 表示该位置温度 T 已知
    # bc_type == 1 表示该位置热流 q 已知
    mask_T_known = (bc_type == 0).float()
    mask_q_known = (bc_type == 1).float()

    # 初始化“已知温度/热流”向量，未知位置先置 0
    T_known = torch.zeros_like(T_true)
    q_known = torch.zeros_like(q_true)

    # 将边界条件值填入相应已知位置
    T_known[mask_T_known > 0.5] = bc_value[mask_T_known > 0.5]
    q_known[mask_q_known > 0.5] = bc_value[mask_q_known > 0.5]

    # 网络输入只提供“已知物理量 + 几何信息”
    # 未知位置保持为 0，由网络自行学习推断
    known_T_feature = T_known.unsqueeze(1)
    known_q_feature = q_known.unsqueeze(1)

    # 构造节点特征
    # 每个节点特征包括：
    # [x, y, nx, ny, len, is_T_known, is_q_known, known_T, known_q]
    x = torch.cat([
        centers,                    # 节点中心坐标 (x, y)
        normals,                    # 法向量 (nx, ny)
        lengths.unsqueeze(1),       # 单元长度
        mask_T_known.unsqueeze(1),  # 是否温度已知
        mask_q_known.unsqueeze(1),  # 是否热流已知
        known_T_feature,            # 已知温度值（未知处为 0）
        known_q_feature,            # 已知热流值（未知处为 0）
    ], dim=1)

    # 根据节点中心构造全连接图
    edge_index, edge_weight = build_full_graph(centers)

    # 封装成 PyTorch Geometric 的 Data 对象
    data = Data(
        x=x,                         # 节点特征
        edge_index=edge_index,       # 边连接关系
        edge_weight=edge_weight,     # 边权重
        centers=centers,             # 单元中心
        normals=normals,             # 单元法向
        lengths=lengths,             # 单元长度
        H_mat=H,                     # BEM H 矩阵
        G_mat=G,                     # BEM G 矩阵
        bc_type=bc_type,             # 边界条件类型
        bc_value=bc_value,           # 边界条件值
        T_known=T_known,             # 已知温度
        q_known=q_known,             # 已知热流
        mask_T_known=mask_T_known,   # 温度已知掩码
        mask_q_known=mask_q_known,   # 热流已知掩码
        T_true=T_true,               # 真实温度
        q_true=q_true,               # 真实热流
        boundary_nodes=boundary_nodes,  # 边界节点
    )
    return data


def save_dataset(root_dir: str = "dataset_rect_bem_gnn"):
    """
    保存数据集到指定目录。

    由于该问题是固定边值问题，不存在多组不同样本，
    因此 train/val/test 三个划分中保存的是同一个图样本。
    """
    root = Path(root_dir)

    # 创建 train / val / test 目录
    for split in ["train", "val", "test"]:
        (root / split).mkdir(parents=True, exist_ok=True)

    # 构建单个图数据
    graph = build_single_graph_data()

    # 这个问题是固定边值问题，所以 train/val/test 都保存同一个图样本
    # 训练本质上是单图上的 physics-informed 拟合
    # 而不是传统意义上的多样本统计学习
    torch.save(graph, root / "train" / "sample_000.pt")
    torch.save(graph, root / "val" / "sample_000.pt")
    torch.save(graph, root / "test" / "sample_000.pt")

    # 输出保存信息
    print(f"Saved dataset to: {root.resolve()}")
    print("train/val/test each contains exactly 1 graph sample.")


if __name__ == "__main__":
    save_dataset()
