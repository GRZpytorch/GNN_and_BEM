import json
from pathlib import Path
from typing import Dict, Tuple
from numpy.polynomial.legendre import leggauss
import numpy as np


class ConstantElementLaplaceBEM:
    """
    2D Laplace 方程常数单元边界元法。

    约定:
        H @ T = G @ q

    其中:
        - T: 边界温度（Dirichlet 量）
        - q: 边界外法向导数 dT/dn（Neumann 量）

    几何说明:
        - 边界为闭合多边形
        - 节点按逆时针顺序给出
        - 单元 j 表示从 node[j] 到 node[j+1] 的边界段
        - 每个单元的配点取该单元中心
    """

    def __init__(self, boundary_nodes: np.ndarray, n_gauss: int = 12):
        """
        初始化边界元对象。

        参数:
            boundary_nodes: 闭合边界节点坐标，形状为 [N, 2]
            n_gauss: 高斯积分点数
        """
        self.boundary_nodes = np.asarray(boundary_nodes, dtype=float)
        self.n_gauss = int(n_gauss)

        # 检查输入是否为二维点坐标数组
        if self.boundary_nodes.ndim != 2 or self.boundary_nodes.shape[1] != 2:
            raise ValueError("boundary_nodes 必须是 [N, 2] 数组")

        # 至少需要 4 个点（例如矩形闭合至少要 5 个点，但这里是泛指最少结构）
        if len(self.boundary_nodes) < 4:
            raise ValueError("boundary_nodes 至少需要 4 个点（含闭合点）")

        # 检查是否闭合：最后一个点必须与第一个点一致
        if not np.allclose(self.boundary_nodes[0], self.boundary_nodes[-1]):
            raise ValueError("boundary_nodes 必须闭合：最后一个点必须等于第一个点")

        # 根据节点构造边界单元几何信息
        self._build_elements()

        # 后续组装/求解得到的矩阵与结果，先初始化为空
        self.H = None
        self.G = None
        self.T = None
        self.q = None

    def _build_elements(self) -> None:
        """
        根据闭合边界节点构造边界单元的几何属性，包括：
            - 单元起点 x1
            - 单元终点 x2
            - 单元长度
            - 切向量
            - 外法向量
            - 单元中心（配点）
        """
        nodes = self.boundary_nodes

        # 单元数 = 节点数 - 1（因为最后一个点是闭合重复点）
        self.n_elem = len(nodes) - 1

        # 每个单元的起点和终点
        self.x1 = nodes[:-1].copy()
        self.x2 = nodes[1:].copy()

        # 单元边向量
        edge_vec = self.x2 - self.x1

        # 单元长度
        self.length = np.linalg.norm(edge_vec, axis=1)
        if np.any(self.length <= 1e-14):
            raise ValueError("存在长度为 0 的边界单元")

        # 单位切向量
        self.tangent = edge_vec / self.length[:, None]

        # 对逆时针边界，外法向可由切向量旋转得到
        # 若切向量为 (tx, ty)，则外法向为 (ty, -tx)
        self.normal = np.column_stack([self.tangent[:, 1], -self.tangent[:, 0]])

        # 配点取每个单元的中心
        self.center = 0.5 * (self.x1 + self.x2)

    @staticmethod
    def fundamental_solution(r: np.ndarray) -> np.ndarray:
        """
        2D Laplace 方程基本解：
            G(r) = -(1 / 2π) * ln(r)

        参数:
            r: 距离

        返回:
            基本解函数值
        """
        return -(1.0 / (2.0 * np.pi)) * np.log(r)

    @staticmethod
    def dG_dn(x_col: np.ndarray, y: np.ndarray, n_y: np.ndarray) -> np.ndarray:
        """
        计算基本解对源点法向的导数 dG/dn_y。

        参数:
            x_col: 配点（观测点）
            y: 源点坐标，可为多个高斯点
            n_y: 源点所在单元的外法向

        返回:
            dG/dn_y 的值
        """
        diff = x_col - y
        r2 = np.sum(diff * diff, axis=-1)

        # 公式:
        # dG/dn_y = (1 / 2π) * ((x - y) · n_y) / |x - y|^2
        return (1.0 / (2.0 * np.pi)) * np.sum(diff * n_y, axis=-1) / r2

    def _gauss_points_on_element(self, j: int):
        """
        生成第 j 个单元上的高斯积分点。

        参数:
            j: 单元编号

        返回:
            y_gp: 高斯点坐标
            wi: 高斯权重
            jac: 从参考区间 [-1, 1] 映射到物理单元的雅可比
        """
        # 在参考区间 [-1, 1] 上获取高斯点和权重
        xi, wi = leggauss(self.n_gauss)

        p1 = self.x1[j]
        p2 = self.x2[j]

        # 将参考区间上的高斯点映射到实际线段单元
        y_gp = 0.5 * ((1.0 - xi)[:, None] * p1 + (1.0 + xi)[:, None] * p2)

        # 线段映射的雅可比
        jac = self.length[j] / 2.0
        return y_gp, wi, jac

    def integrate_G(self, i: int, j: int) -> float:
        """
        计算 G 矩阵中第 (i, j) 项：
            G_ij = ∫_{Γ_j} G(x_i, y) dΓ_y

        参数:
            i: 配点编号
            j: 单元编号

        返回:
            积分结果
        """
        # 对角项存在对数弱奇异性，这里使用常数单元解析积分公式
        if i == j:
            L = self.length[j]
            return -(L / (2.0 * np.pi)) * (np.log(L / 2.0) - 1.0)

        # 非对角项使用高斯积分
        x_col = self.center[i]
        y_gp, wi, jac = self._gauss_points_on_element(j)
        r = np.linalg.norm(x_col - y_gp, axis=1)
        val = self.fundamental_solution(r)
        return float(np.sum(val * wi) * jac)

    def integrate_H_offdiag(self, i: int, j: int) -> float:
        """
        计算 H 矩阵非对角项：
            H_ij = ∫_{Γ_j} dG/dn_y (x_i, y) dΓ_y, i != j

        参数:
            i: 配点编号
            j: 单元编号

        返回:
            积分结果
        """
        x_col = self.center[i]
        y_gp, wi, jac = self._gauss_points_on_element(j)
        val = self.dG_dn(x_col, y_gp, self.normal[j])
        return float(np.sum(val * wi) * jac)

    def assemble_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        组装边界元系统矩阵 H 和 G。

        返回:
            H, G
        """
        n = self.n_elem
        H = np.zeros((n, n), dtype=float)
        G = np.zeros((n, n), dtype=float)

        # 逐项计算矩阵元素
        for i in range(n):
            for j in range(n):
                G[i, j] = self.integrate_G(i, j)

                # H 的非对角项通过数值积分得到
                if i != j:
                    H[i, j] = self.integrate_H_offdiag(i, j)

        # 对角项通过 row-sum 修正得到
        # 这样可以增强常数解的稳定性
        for i in range(n):
            H[i, i] = -np.sum(H[i, :])

        self.H = H
        self.G = G
        return H, G

    def solve(self, bc_type: np.ndarray, bc_value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据边界条件求解边界上的 T 和 q。

        参数:
            bc_type:
                边界条件类型数组
                - 0: Dirichlet（温度 T 已知）
                - 1: Neumann（热流 q 已知）
            bc_value:
                对应边界条件的数值

        返回:
            T: 边界温度
            q: 边界外法向导数
        """
        # 若矩阵尚未组装，则先组装
        if self.H is None or self.G is None:
            self.assemble_matrices()

        bc_type = np.asarray(bc_type)
        bc_value = np.asarray(bc_value, dtype=float)
        n = self.n_elem

        # 检查边界条件长度是否与单元数一致
        if bc_type.shape != (n,) or bc_value.shape != (n,):
            raise ValueError("bc_type 和 bc_value 的长度必须等于单元数")

        # 找出 Dirichlet 与 Neumann 单元索引
        dir_idx = np.where(bc_type == 0)[0]
        neu_idx = np.where(bc_type == 1)[0]

        if len(dir_idx) + len(neu_idx) != n:
            raise ValueError("bc_type 必须只包含 0(Dirichlet) 或 1(Neumann)")

        # 已知量初始化
        T_known = np.zeros(n, dtype=float)
        q_known = np.zeros(n, dtype=float)

        # 按边界条件填入已知值
        T_known[dir_idx] = bc_value[dir_idx]
        q_known[neu_idx] = bc_value[neu_idx]

        # 构造线性方程组 A z = b
        # 未知量顺序为：
        #   z = [T[neu_idx], q[dir_idx]]
        A = np.hstack([
            self.H[:, neu_idx],     # 对未知 T 的系数
            -self.G[:, dir_idx],    # 对未知 q 的系数
        ])

        b = self.G[:, neu_idx] @ q_known[neu_idx] - self.H[:, dir_idx] @ T_known[dir_idx]

        # 求解未知边界量
        z = np.linalg.solve(A, b)

        # 恢复完整 T 和 q
        T = np.zeros(n, dtype=float)
        q = np.zeros(n, dtype=float)

        T[dir_idx] = T_known[dir_idx]
        q[neu_idx] = q_known[neu_idx]

        T[neu_idx] = z[: len(neu_idx)]
        q[dir_idx] = z[len(neu_idx):]

        self.T = T
        self.q = q
        return T, q

    def evaluate_interior(self, points: np.ndarray) -> np.ndarray:
        """
        计算内部点的温度值。

        参数:
            points: 内部点坐标数组，形状为 [M, 2]

        返回:
            内部点温度值数组
        """
        if self.T is None or self.q is None:
            raise RuntimeError("请先调用 solve()")

        points = np.asarray(points, dtype=float)
        out = np.zeros(len(points), dtype=float)

        # 对每个内部点分别进行边界积分
        for m, x in enumerate(points):
            val = 0.0
            for j in range(self.n_elem):
                y_gp, wi, jac = self._gauss_points_on_element(j)
                r = np.linalg.norm(x - y_gp, axis=1)

                # 基本解与法向导数
                Gv = self.fundamental_solution(r)
                Hv = self.dG_dn(x, y_gp, self.normal[j])

                # 边界积分表达式:
                # T(x) = ∫ q(y) G(x,y) dΓ - ∫ T(y) dG/dn_y dΓ
                val += np.sum(self.q[j] * Gv * wi) * jac
                val -= np.sum(self.T[j] * Hv * wi) * jac

            out[m] = val

        return out


def build_rectangle_boundary(width: float = 2.0, height: float = 1.0,
                             n_bottom: int = 16, n_right: int = 8,
                             n_top: int = 16, n_left: int = 8) -> np.ndarray:
    """
    构造矩形边界节点。

    参数:
        width: 矩形宽度
        height: 矩形高度
        n_bottom: 下边离散单元数
        n_right: 右边离散单元数
        n_top: 上边离散单元数
        n_left: 左边离散单元数

    返回:
        闭合边界节点序列，按逆时针排列
    """
    # 下边：从左到右
    bottom = np.column_stack([
        np.linspace(0.0, width, n_bottom + 1),
        np.zeros(n_bottom + 1),
    ])

    # 右边：从下到上
    right = np.column_stack([
        np.full(n_right + 1, width),
        np.linspace(0.0, height, n_right + 1),
    ])

    # 上边：从右到左
    top = np.column_stack([
        np.linspace(width, 0.0, n_top + 1),
        np.full(n_top + 1, height),
    ])

    # 左边：从上到下
    left = np.column_stack([
        np.zeros(n_left + 1),
        np.linspace(height, 0.0, n_left + 1),
    ])

    # 去掉前 3 条边最后一个重复点，保留最终闭合
    return np.vstack([bottom[:-1], right[:-1], top[:-1], left])


def build_problem_definition(n_bottom: int = 16, n_right: int = 8,
                             n_top: int = 16, n_left: int = 8) -> Dict[str, np.ndarray]:
    """
    构建一个矩形区域上的边界值问题定义。

    问题设置:
        - 左边界: Dirichlet = 1
        - 右边界: Dirichlet = 2
        - 上边界: Neumann = 0
        - 下边界: Neumann = 0

    返回:
        包含几何信息、边界条件、BEM 矩阵及真值解的字典
    """
    # 构造矩形边界
    boundary_nodes = build_rectangle_boundary(
        width=2.0,
        height=1.0,
        n_bottom=n_bottom,
        n_right=n_right,
        n_top=n_top,
        n_left=n_left,
    )

    # 创建 BEM 对象并组装矩阵
    bem = ConstantElementLaplaceBEM(boundary_nodes, n_gauss=12)
    bem.assemble_matrices()
    n = bem.n_elem

    # 初始化边界条件类型和值
    bc_type = np.empty(n, dtype=np.int64)
    bc_value = np.zeros(n, dtype=float)

    # 各边对应的单元编号范围
    idx_bottom = np.arange(0, n_bottom)
    idx_right = np.arange(n_bottom, n_bottom + n_right)
    idx_top = np.arange(n_bottom + n_right, n_bottom + n_right + n_top)
    idx_left = np.arange(n_bottom + n_right + n_top, n)

    # 左边界：Dirichlet = 1
    bc_type[idx_left] = 0
    bc_value[idx_left] = 1.0

    # 右边界：Dirichlet = 2
    bc_type[idx_right] = 0
    bc_value[idx_right] = 2.0

    # 下边界：Neumann = 0
    bc_type[idx_bottom] = 1
    bc_value[idx_bottom] = 0.0

    # 上边界：Neumann = 0
    bc_type[idx_top] = 1
    bc_value[idx_top] = 0.0

    # 求解真实边界解
    T, q = bem.solve(bc_type, bc_value)

    return {
        "boundary_nodes": boundary_nodes,  # 边界节点
        "centers": bem.center,             # 单元中心
        "normals": bem.normal,             # 外法向
        "lengths": bem.length,             # 单元长度
        "H": bem.H,                        # H 矩阵
        "G": bem.G,                        # G 矩阵
        "bc_type": bc_type,                # 边界条件类型
        "bc_value": bc_value,              # 边界条件值
        "T_true": T,                       # 真实温度解
        "q_true": q,                       # 真实热流解
        "n_bottom": np.array([n_bottom], dtype=np.int64),
        "n_right": np.array([n_right], dtype=np.int64),
        "n_top": np.array([n_top], dtype=np.int64),
        "n_left": np.array([n_left], dtype=np.int64),
    }


def save_problem_preview(out_dir: str = "artifacts/problem_preview") -> None:
    """
    保存问题预览信息到 JSON 文件中，便于快速查看问题规模与解的范围。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = build_problem_definition()

    # 组织预览信息
    preview = {
        "n_elem": int(len(data["centers"])),  # 单元总数
        "bc_type_counts": {
            "Dirichlet": int(np.sum(data["bc_type"] == 0)),
            "Neumann": int(np.sum(data["bc_type"] == 1)),
        },
        "T_true_min": float(data["T_true"].min()),
        "T_true_max": float(data["T_true"].max()),
        "q_true_min": float(data["q_true"].min()),
        "q_true_max": float(data["q_true"].max()),
    }

    # 写入 JSON 文件
    with open(out_path / "preview.json", "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 构建问题并输出基本信息
    data = build_problem_definition()
    print("n_elem =", len(data["centers"]))
    print("T_true range =", data["T_true"].min(), data["T_true"].max())
    print("q_true range =", data["q_true"].min(), data["q_true"].max())

    # 保存问题预览
    save_problem_preview()
