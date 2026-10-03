"""Constant-element BEM solver for the two-dimensional Laplace reference problem."""

import json
from pathlib import Path
from typing import Dict, Tuple
from numpy.polynomial.legendre import leggauss
import numpy as np

class ConstantElementLaplaceBEM:

    def __init__(self, boundary_nodes: np.ndarray, n_gauss: int=12):
        self.boundary_nodes = np.asarray(boundary_nodes, dtype=float)
        self.n_gauss = int(n_gauss)
        if self.boundary_nodes.ndim != 2 or self.boundary_nodes.shape[1] != 2:
            raise ValueError('Runtime message')
        if len(self.boundary_nodes) < 4:
            raise ValueError('Runtime message')
        if not np.allclose(self.boundary_nodes[0], self.boundary_nodes[-1]):
            raise ValueError('Runtime message')
        self._build_elements()
        self.H = None
        self.G = None
        self.T = None
        self.q = None

    def _build_elements(self) -> None:
        nodes = self.boundary_nodes
        self.n_elem = len(nodes) - 1
        self.x1 = nodes[:-1].copy()
        self.x2 = nodes[1:].copy()
        edge_vec = self.x2 - self.x1
        self.length = np.linalg.norm(edge_vec, axis=1)
        if np.any(self.length <= 1e-14):
            raise ValueError('Runtime message')
        self.tangent = edge_vec / self.length[:, None]
        self.normal = np.column_stack([self.tangent[:, 1], -self.tangent[:, 0]])
        self.center = 0.5 * (self.x1 + self.x2)

    @staticmethod
    def fundamental_solution(r: np.ndarray) -> np.ndarray:
        return -(1.0 / (2.0 * np.pi)) * np.log(r)

    @staticmethod
    def dG_dn(x_col: np.ndarray, y: np.ndarray, n_y: np.ndarray) -> np.ndarray:
        diff = x_col - y
        r2 = np.sum(diff * diff, axis=-1)
        return 1.0 / (2.0 * np.pi) * np.sum(diff * n_y, axis=-1) / r2

    def _gauss_points_on_element(self, j: int):
        xi, wi = leggauss(self.n_gauss)
        p1 = self.x1[j]
        p2 = self.x2[j]
        y_gp = 0.5 * ((1.0 - xi)[:, None] * p1 + (1.0 + xi)[:, None] * p2)
        jac = self.length[j] / 2.0
        return (y_gp, wi, jac)

    def integrate_G(self, i: int, j: int) -> float:
        if i == j:
            L = self.length[j]
            return -(L / (2.0 * np.pi)) * (np.log(L / 2.0) - 1.0)
        x_col = self.center[i]
        y_gp, wi, jac = self._gauss_points_on_element(j)
        r = np.linalg.norm(x_col - y_gp, axis=1)
        val = self.fundamental_solution(r)
        return float(np.sum(val * wi) * jac)

    def integrate_H_offdiag(self, i: int, j: int) -> float:
        x_col = self.center[i]
        y_gp, wi, jac = self._gauss_points_on_element(j)
        val = self.dG_dn(x_col, y_gp, self.normal[j])
        return float(np.sum(val * wi) * jac)

    def assemble_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_elem
        H = np.zeros((n, n), dtype=float)
        G = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                G[i, j] = self.integrate_G(i, j)
                if i != j:
                    H[i, j] = self.integrate_H_offdiag(i, j)
        for i in range(n):
            H[i, i] = -np.sum(H[i, :])
        self.H = H
        self.G = G
        return (H, G)

    def solve(self, bc_type: np.ndarray, bc_value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.H is None or self.G is None:
            self.assemble_matrices()
        bc_type = np.asarray(bc_type)
        bc_value = np.asarray(bc_value, dtype=float)
        n = self.n_elem
        if bc_type.shape != (n,) or bc_value.shape != (n,):
            raise ValueError('bc_type and bc_value must have one entry per boundary element.')
        dir_idx = np.where(bc_type == 0)[0]
        neu_idx = np.where(bc_type == 1)[0]
        if len(dir_idx) + len(neu_idx) != n:
            raise ValueError('bc_type must contain only 0 (Dirichlet) or 1 (Neumann).')
        T_known = np.zeros(n, dtype=float)
        q_known = np.zeros(n, dtype=float)
        T_known[dir_idx] = bc_value[dir_idx]
        q_known[neu_idx] = bc_value[neu_idx]
        A = np.hstack([self.H[:, neu_idx], -self.G[:, dir_idx]])
        b = self.G[:, neu_idx] @ q_known[neu_idx] - self.H[:, dir_idx] @ T_known[dir_idx]
        z = np.linalg.solve(A, b)
        T = np.zeros(n, dtype=float)
        q = np.zeros(n, dtype=float)
        T[dir_idx] = T_known[dir_idx]
        q[neu_idx] = q_known[neu_idx]
        T[neu_idx] = z[:len(neu_idx)]
        q[dir_idx] = z[len(neu_idx):]
        self.T = T
        self.q = q
        return (T, q)

    def evaluate_interior(self, points: np.ndarray) -> np.ndarray:
        if self.T is None or self.q is None:
            raise RuntimeError('Call solve() before evaluating the interior field.')
        points = np.asarray(points, dtype=float)
        out = np.zeros(len(points), dtype=float)
        for m, x in enumerate(points):
            val = 0.0
            for j in range(self.n_elem):
                y_gp, wi, jac = self._gauss_points_on_element(j)
                r = np.linalg.norm(x - y_gp, axis=1)
                Gv = self.fundamental_solution(r)
                Hv = self.dG_dn(x, y_gp, self.normal[j])
                val += np.sum(self.q[j] * Gv * wi) * jac
                val -= np.sum(self.T[j] * Hv * wi) * jac
            out[m] = val
        return out

def build_rectangle_boundary(width: float=2.0, height: float=1.0, n_bottom: int=16, n_right: int=8, n_top: int=16, n_left: int=8) -> np.ndarray:
    bottom = np.column_stack([np.linspace(0.0, width, n_bottom + 1), np.zeros(n_bottom + 1)])
    right = np.column_stack([np.full(n_right + 1, width), np.linspace(0.0, height, n_right + 1)])
    top = np.column_stack([np.linspace(width, 0.0, n_top + 1), np.full(n_top + 1, height)])
    left = np.column_stack([np.zeros(n_left + 1), np.linspace(height, 0.0, n_left + 1)])
    return np.vstack([bottom[:-1], right[:-1], top[:-1], left])

def build_problem_definition(n_bottom: int=16, n_right: int=8, n_top: int=16, n_left: int=8) -> Dict[str, np.ndarray]:
    boundary_nodes = build_rectangle_boundary(width=2.0, height=1.0, n_bottom=n_bottom, n_right=n_right, n_top=n_top, n_left=n_left)
    bem = ConstantElementLaplaceBEM(boundary_nodes, n_gauss=12)
    bem.assemble_matrices()
    n = bem.n_elem
    bc_type = np.empty(n, dtype=np.int64)
    bc_value = np.zeros(n, dtype=float)
    idx_bottom = np.arange(0, n_bottom)
    idx_right = np.arange(n_bottom, n_bottom + n_right)
    idx_top = np.arange(n_bottom + n_right, n_bottom + n_right + n_top)
    idx_left = np.arange(n_bottom + n_right + n_top, n)
    bc_type[idx_left] = 0
    bc_value[idx_left] = 1.0
    bc_type[idx_right] = 0
    bc_value[idx_right] = 2.0
    bc_type[idx_bottom] = 1
    bc_value[idx_bottom] = 0.0
    bc_type[idx_top] = 1
    bc_value[idx_top] = 0.0
    T, q = bem.solve(bc_type, bc_value)
    return {'boundary_nodes': boundary_nodes, 'centers': bem.center, 'normals': bem.normal, 'lengths': bem.length, 'H': bem.H, 'G': bem.G, 'bc_type': bc_type, 'bc_value': bc_value, 'T_true': T, 'q_true': q, 'n_bottom': np.array([n_bottom], dtype=np.int64), 'n_right': np.array([n_right], dtype=np.int64), 'n_top': np.array([n_top], dtype=np.int64), 'n_left': np.array([n_left], dtype=np.int64)}

def save_problem_preview(out_dir: str='artifacts/problem_preview') -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data = build_problem_definition()
    preview = {'n_elem': int(len(data['centers'])), 'bc_type_counts': {'Dirichlet': int(np.sum(data['bc_type'] == 0)), 'Neumann': int(np.sum(data['bc_type'] == 1))}, 'T_true_min': float(data['T_true'].min()), 'T_true_max': float(data['T_true'].max()), 'q_true_min': float(data['q_true'].min()), 'q_true_max': float(data['q_true'].max())}
    with open(out_path / 'preview.json', 'w', encoding='utf-8') as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    data = build_problem_definition()
    print('n_elem =', len(data['centers']))
    print('T_true range =', data['T_true'].min(), data['T_true'].max())
    print('q_true range =', data['q_true'].min(), data['q_true'].max())
    save_problem_preview()
