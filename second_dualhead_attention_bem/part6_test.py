import json
from pathlib import Path
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from part1_bem_solver import ConstantElementLaplaceBEM
from part3_dataset import build_loaders
from part4_model import DualHeadGCNVAE, assemble_full_fields


# ============================================================
# 全局绘图配置
# 以后如果你想改样式，优先改这里，不要到处改函数内部
# ============================================================

@dataclass
class PlotConfig:
    # ---------- 字体 ----------
    font_family: str = "Times New Roman"
    title_fontsize: int = 23         # 标题字号
    label_fontsize: int = 27         # 坐标轴标签字号
    tick_fontsize: int = 21          # 刻度字号
    label_fontstyle: str = "italic"  # 新增：坐标轴标签斜体
    legend_fontsize: int = 21        # 图例字号
    colorbar_fontsize: int = 13     # 色条刻度字号

    # ---------- 标题、标签、刻度与坐标轴的距离 ----------
    title_pad: float = 7.0           # 标题距离子图的间距
    xlabel_pad: float = 2.0          # x标签与坐标轴间距
    ylabel_pad: float = 2.0          # y标签与坐标轴间距
    tick_pad: float = 4.0            # 刻度数字与坐标轴间距

    # ---------- 画布大小 ----------
    fig1_size: tuple = (24.0, 6)     # 图1大小
    fig23_size: tuple = (36.0, 6)    # 图2/图3大小
    fig4_size: tuple = (36.0, 6)     # 图4大小

    # ---------- 图片保存 ----------
    save_dpi: int = 900              # png分辨率

    # ---------- 散点样式 ----------
    scatter_size: float = 128.0       # 普通云图点大小
    marker_edge_width: float = 0.40  # 普通点黑边线宽

    # 未知点标记：改细一点，不那么显眼
    unknown_circle_size: float = 128.0
    unknown_cross_size: float = 36.0
    unknown_marker_width: float = 0.90

    # ---------- 坐标轴样式 ----------
    axis_label_x: str = "x"
    axis_label_y: str = "y"
    axis_equal: bool = True          # 是否保持x/y等比例

    show_grid: bool = False
    grid_alpha: float = 0.25

    grid_linestyle: str = "--"
    axes_linewidth: float = 1.0

    # ---------- 刻度样式 ----------
    tick_width: float = 1.0
    tick_length: float = 4.0

    # ---------- 坐标轴范围 ----------
    xlim: tuple | None = (-0.55, 2.55)
    ylim: tuple | None = (-0.25, 1.25)

    # ---------- 色条样式 ----------
    colorbar_fraction: float = 0.050   # 色条宽度
    colorbar_pad: float = 0.015        # 色条距离图的距离
    colorbar_shrink: float = 0.95      # 色条高度
    colorbar_aspect: float = 28        # 色条粗细
    colorbar_n_ticks: int = 7          # 统一7个刻度（包含最小值和最大值）

    # ---------- 布局 ----------
    use_tight_layout: bool = True

    # ---------- 域内云图分辨率 ----------
    interior_nx: int = 120
    interior_ny: int = 60
    image_interpolation: str = "bilinear"

    # ---------- 配色 ----------
    cmap_dirichlet: str = "viridis"
    cmap_neumann: str = "plasma"
    cmap_interior: str = "turbo"
    cmap_error: str = "coolwarm"

    # ---------- 图标题 ----------
    title_fig1_left: str = "Known Dirichlet Boundary Conditions"
    title_fig1_right: str = "Known Neumann Boundary Conditions"

    title_dirichlet_true: str = "Complete Dirichlet Boundary (True)"
    title_dirichlet_pred: str = "Complete Dirichlet Boundary (Predicted)"
    title_dirichlet_err: str = "Dirichlet Boundary Error "

    title_neumann_true: str = "Complete Neumann Boundary (True)"
    title_neumann_pred: str = "Complete Neumann Boundary (Predicted)"
    title_neumann_err: str = "Neumann Boundary Error "

    title_interior_true: str = "Interior Field from True Boundary"
    title_interior_pred: str = "Interior Field from Predicted Boundary"
    title_interior_err: str = "Interior Field Error "

    # ---------- 图例 ----------
    legend_unknown_circle: str = "Unknown location"
    legend_unknown_cross: str = "Unknown marker"


# 创建一个全局配置实例
PLOT_CFG = PlotConfig()


def apply_plot_config(cfg: PlotConfig):
    """
    应用全局 matplotlib 样式。
    """
    matplotlib.rcParams["font.family"] = cfg.font_family
    matplotlib.rcParams["font.size"] = cfg.label_fontsize
    matplotlib.rcParams["axes.titlesize"] = cfg.title_fontsize
    matplotlib.rcParams["axes.labelsize"] = cfg.label_fontsize
    matplotlib.rcParams["xtick.labelsize"] = cfg.tick_fontsize
    matplotlib.rcParams["ytick.labelsize"] = cfg.tick_fontsize
    matplotlib.rcParams["legend.fontsize"] = cfg.legend_fontsize
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["axes.linewidth"] = cfg.axes_linewidth


apply_plot_config(PLOT_CFG)


# ============================================================
# 基础工具函数
# ============================================================

def _to_numpy(x):
    """
    将 torch.Tensor 或其他数组转换为 numpy 数组。
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _make_unknown_legend(cfg=PLOT_CFG):
    """
    构造图1的图例：
    1. 已知部分：半黄半青色填充圆点 (代表云图中间色)
    2. 未知部分：红色叉号
    """
    # 定义“半黄半青”的颜色 (RGBA 或是特定的十六进制)
    # 这个颜色接近 viridis 的中间色调
    mid_color = "#3cb371"  # 中绿色/青黄色调，或者使用 (0.5, 0.7, 0.4)

    return [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=mid_color,
            markeredgecolor="k",
            markeredgewidth=cfg.marker_edge_width,
            markersize=8,
            label="Known",
        ),
        Line2D(
            [0], [0],
            marker="x",
            color="red",
            linestyle="None",
            markersize=8,
            markeredgewidth=cfg.unknown_marker_width,
            label="Unknown",
        ),
    ]




def _style_axis(ax, cfg=PLOT_CFG):
    """
    统一设置坐标轴样式：
    - 标签
    - 标签与轴间距
    - 刻度方向向内
    - 坐标轴范围
    - 网格
    """
    ax.set_xlabel(cfg.axis_label_x, fontsize=cfg.label_fontsize, labelpad=cfg.xlabel_pad,fontstyle=cfg.label_fontstyle)
    ax.set_ylabel(cfg.axis_label_y, fontsize=cfg.label_fontsize, labelpad=cfg.ylabel_pad,fontstyle=cfg.label_fontstyle)

    if cfg.axis_equal:
        ax.set_aspect("equal")

    ax.grid(cfg.show_grid)

    # 横纵坐标刻度保留2位小数
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=cfg.tick_fontsize,
        width=cfg.tick_width,
        length=cfg.tick_length,
        pad=cfg.tick_pad,
        direction="in",
    )

    if cfg.xlim is not None:
        ax.set_xlim(cfg.xlim)
    if cfg.ylim is not None:
        ax.set_ylim(cfg.ylim)



def _save_figure(fig, path, cfg=PLOT_CFG):
    """
    同时保存为：
    - PNG（位图）
    - SVG（矢量图）

    path 请传入 .png 路径。
    """
    fig.savefig(path, dpi=cfg.save_dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")

def _expand_constant_range(vmin, vmax, delta=0.003):
    """
    若 vmin 和 vmax 相等（或几乎相等），
    则人为扩展成 [c-delta, c+delta]，便于色条显示。
    """
    if vmin is None or vmax is None:
        return vmin, vmax

    vmin = float(vmin)
    vmax = float(vmax)

    if np.isclose(vmin, vmax):
        c = 0.5 * (vmin + vmax)
        return c - delta, c + delta

    return vmin, vmax

def _make_colorbar_ticks(vmin, vmax, n_ticks=7):
    """
    生成统一色条刻度：
    - 总共 n_ticks 个
    - 一定包含最小值和最大值
    """
    if vmin is None or vmax is None:
        return None

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None

    if np.isclose(vmin, vmax):
        eps = max(abs(vmin) * 1e-6, 1e-12)
        return np.linspace(vmin - 3 * eps, vmax + 3 * eps, n_ticks)

    return np.linspace(vmin, vmax, n_ticks)


def _format_colorbar(cbar, vmin, vmax, cfg=PLOT_CFG):
    """
    统一设置色条：
    - 共 cfg.colorbar_n_ticks 个刻度
    - 包含最小值和最大值
    - 刻度保留三位小数
    """
    ticks = _make_colorbar_ticks(vmin, vmax, cfg.colorbar_n_ticks)
    if ticks is not None:
        cbar.set_ticks(ticks)

    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    cbar.ax.tick_params(
        labelsize=cfg.colorbar_fontsize,
        direction="in",
    )




def _create_colorbar(mappable, ax, vmin, vmax, cfg=PLOT_CFG):
    """
    创建并统一格式化色条。
    """
    cbar = plt.colorbar(
        mappable,
        ax=ax,
        fraction=cfg.colorbar_fraction,
        pad=cfg.colorbar_pad,
        shrink=cfg.colorbar_shrink,
        aspect=cfg.colorbar_aspect,
    )
    _format_colorbar(cbar, vmin, vmax, cfg)
    return cbar


# ============================================================
# 边界云图工具
# ============================================================

def _scatter_known_unknown(
    ax,
    xy,
    values,
    known_mask,
    unknown_mask,
    title,
    cmap=None,
    vmin=None,
    vmax=None,
    cfg=PLOT_CFG,
):
    """
    已知值画彩色散点；未知值在图中画红圈+红叉，但图例只显示红叉。
    """
    """
    已知值画彩色散点；
    未知值画红色空心圆 + 红色 x。
    """
    xy_np = _to_numpy(xy)
    values_np = _to_numpy(values)
    known_mask_np = _to_numpy(known_mask).astype(bool)
    unknown_mask_np = _to_numpy(unknown_mask).astype(bool)

    x = xy_np[:, 0]
    y = xy_np[:, 1]

    sc = None
    if np.any(known_mask_np):
        sc = ax.scatter(
            x[known_mask_np],
            y[known_mask_np],
            c=values_np[known_mask_np],
            s=cfg.scatter_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="k",
            linewidths=cfg.marker_edge_width,
        )

    if np.any(unknown_mask_np):
        ax.scatter(
            x[unknown_mask_np],
            y[unknown_mask_np],
            s=cfg.unknown_circle_size,
            facecolors="none",
            edgecolors="red",
            linewidths=cfg.unknown_marker_width,
            zorder=4,
        )
        ax.scatter(
            x[unknown_mask_np],
            y[unknown_mask_np],
            s=cfg.unknown_cross_size,
            c="red",
            marker="x",
            linewidths=cfg.unknown_marker_width,
            zorder=5,
        )

    ax.set_title(title, fontsize=cfg.title_fontsize, pad=cfg.title_pad)
    _style_axis(ax, cfg)

    # 获取包含元组的 handles
    handles = _make_unknown_legend(cfg)

    ax.legend(
        handles=_make_unknown_legend(cfg),
        loc="upper center",  # 核心：位置设为上方居中
        bbox_to_anchor=(0.75, 1.02),  # 锚点设在轴的水平 0.5 处
        ncol=2,  # 核心：设置为 2 列，让已知和未知并排居中
        frameon=True,
        fontsize=cfg.legend_fontsize,
        columnspacing=1.0,  # 调整两项之间的间距
        handletextpad=0.5  # 调整图标与文字之间的间距
    )

    return sc


def _scatter_field(
    ax,
    xy,
    values,
    title,
    cmap,
    vmin=None,
    vmax=None,
    cfg=PLOT_CFG,
):
    """
    绘制完整边界场云图。
    """
    xy_np = _to_numpy(xy)
    values_np = _to_numpy(values)

    sc = ax.scatter(
        xy_np[:, 0],
        xy_np[:, 1],
        c=values_np,
        s=cfg.scatter_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="k",
        linewidths=cfg.marker_edge_width,
    )
    ax.set_title(title, fontsize=cfg.title_fontsize, pad=cfg.title_pad)
    _style_axis(ax, cfg)
    return sc


def _scatter_error(
    ax,
    xy,
    error,
    title,
    cmap=None,
    cfg=PLOT_CFG,
):
    """
    绘制误差云图，色轴关于0对称。
    """
    xy_np = _to_numpy(xy)
    err_np = _to_numpy(error)
    emax = max(np.max(np.abs(err_np)), 1e-12)

    sc = ax.scatter(
        xy_np[:, 0],
        xy_np[:, 1],
        c=err_np,
        s=cfg.scatter_size,
        cmap=cmap,
        vmin=-emax,
        vmax=emax,
        edgecolors="k",
        linewidths=cfg.marker_edge_width,
    )
    ax.set_title(title, fontsize=cfg.title_fontsize, pad=cfg.title_pad)
    _style_axis(ax, cfg)
    return sc


# ============================================================
# 域内值计算工具
# ============================================================

def _infer_grid(boundary_nodes, nx=120, ny=60):
    """
    根据边界范围自动生成域内规则网格。
    """
    xmin, xmax = boundary_nodes[:, 0].min(), boundary_nodes[:, 0].max()
    ymin, ymax = boundary_nodes[:, 1].min(), boundary_nodes[:, 1].max()

    dx = xmax - xmin
    dy = ymax - ymin

    eps_x = 1e-3 * dx
    eps_y = 1e-3 * dy

    xs = np.linspace(xmin + eps_x, xmax - eps_x, nx)
    ys = np.linspace(ymin + eps_y, ymax - eps_y, ny)

    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    return X, Y, pts


def _compute_interior(bem, T, q, pts, nx, ny):
    """
    用边界上的 T、q 计算域内场，并恢复成 ny × nx 的网格。
    """
    bem.T = T
    bem.q = q
    U = bem.evaluate_interior(pts)
    return U.reshape(ny, nx)


# ============================================================
# 图4：域内值对比图
# ============================================================

def _plot_interior_compare(boundary_nodes, T_pred, q_pred, T_true, q_true, save_path, cfg=PLOT_CFG):
    """
    第四张图：
    - 子图1：真实边界得到的域内值
    - 子图2：预测边界得到的域内值
    - 子图3：域内误差
    """
    boundary_nodes_np = _to_numpy(boundary_nodes)
    T_pred_np = _to_numpy(T_pred)
    q_pred_np = _to_numpy(q_pred)
    T_true_np = _to_numpy(T_true)
    q_true_np = _to_numpy(q_true)

    bem = ConstantElementLaplaceBEM(boundary_nodes_np)
    bem.assemble_matrices()

    nx, ny = cfg.interior_nx, cfg.interior_ny
    X, Y, pts = _infer_grid(boundary_nodes_np, nx, ny)

    U_pred = _compute_interior(bem, T_pred_np, q_pred_np, pts, nx, ny)
    U_true = _compute_interior(bem, T_true_np, q_true_np, pts, nx, ny)
    U_err = U_pred - U_true

    vmin = min(U_pred.min(), U_true.min())
    vmax = max(U_pred.max(), U_true.max())
    err_abs = max(float(np.max(np.abs(U_err))), 1e-12)

    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    fig, axes = plt.subplots(1, 3, figsize=cfg.fig4_size)

    im1 = axes[0].imshow(
        U_true,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal",
        cmap=cfg.cmap_interior,
        vmin=vmin,
        vmax=vmax,
        interpolation=cfg.image_interpolation,
    )
    axes[0].set_title(cfg.title_interior_true, fontsize=cfg.title_fontsize, pad=cfg.title_pad)
    _style_axis(axes[0], cfg)
    _create_colorbar(im1, axes[0], vmin, vmax, cfg)

    im2 = axes[1].imshow(
        U_pred,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal",
        cmap=cfg.cmap_interior,
        vmin=vmin,
        vmax=vmax,
        interpolation=cfg.image_interpolation,
    )
    axes[1].set_title(cfg.title_interior_pred, fontsize=cfg.title_fontsize, pad=cfg.title_pad)
    _style_axis(axes[1], cfg)
    _create_colorbar(im2, axes[1], vmin, vmax, cfg)

    im3 = axes[2].imshow(
        U_err,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal",
        cmap=cfg.cmap_error,
        vmin=-err_abs,
        vmax=err_abs,
        interpolation=cfg.image_interpolation,
    )
    axes[2].set_title(cfg.title_interior_err, fontsize=cfg.title_fontsize, pad=cfg.title_pad)
    _style_axis(axes[2], cfg)
    _create_colorbar(im3, axes[2], -err_abs, err_abs, cfg)

    if cfg.use_tight_layout:
        plt.tight_layout()

    _save_figure(fig, save_path, cfg)
    plt.close(fig)

    return U_pred, U_true, U_err


# ============================================================
# 图1：已知条件云图
# ============================================================

def _plot_known_condition_clouds(data, out_path, cfg=PLOT_CFG):
    """
    第一张图：
    - 子图1：已知 Dirichlet 条件
    - 子图2：已知 Neumann 条件
    未知位置用红圈 + 红色x标出来

    特殊处理：
    - 若子图2（Neumann）已知值全为常数，则色条范围扩展为 c±0.003
    - 这样在保留三位小数时，仍能正常显示 7 个刻度
    """
    xy = data.centers

    mask_T_known = data.mask_T_known > 0.5
    mask_q_known = data.mask_q_known > 0.5

    T_input = torch.full_like(data.T_true, float("nan"))
    q_input = torch.full_like(data.q_true, float("nan"))

    T_input[mask_T_known] = data.T_known[mask_T_known]
    q_input[mask_q_known] = data.q_known[mask_q_known]

    T_known_vals = data.T_known[mask_T_known]
    q_known_vals = data.q_known[mask_q_known]

    T_vmin = float(T_known_vals.min().cpu()) if torch.any(mask_T_known) else None
    T_vmax = float(T_known_vals.max().cpu()) if torch.any(mask_T_known) else None
    q_vmin = float(q_known_vals.min().cpu()) if torch.any(mask_q_known) else None
    q_vmax = float(q_known_vals.max().cpu()) if torch.any(mask_q_known) else None

    # 图1子图2：若已知 Neumann 值全为常数，则扩展为 c±0.003
    q_vmin, q_vmax = _expand_constant_range(q_vmin, q_vmax, delta=0.003)

    fig, axes = plt.subplots(1, 2, figsize=cfg.fig1_size)

    sc1 = _scatter_known_unknown(
        axes[0],
        xy,
        T_input,
        known_mask=mask_T_known,
        unknown_mask=~mask_T_known,
        title=cfg.title_fig1_left,
        cmap=cfg.cmap_dirichlet,
        vmin=T_vmin,
        vmax=T_vmax,
        cfg=cfg,
    )
    if sc1 is not None:
        _create_colorbar(sc1, axes[0], T_vmin, T_vmax, cfg)

    sc2 = _scatter_known_unknown(
        axes[1],
        xy,
        q_input,
        known_mask=mask_q_known,
        unknown_mask=~mask_q_known,
        title=cfg.title_fig1_right,
        cmap=cfg.cmap_neumann,
        vmin=q_vmin,
        vmax=q_vmax,
        cfg=cfg,
    )
    if sc2 is not None:
        _create_colorbar(sc2, axes[1], q_vmin, q_vmax, cfg)

    if cfg.use_tight_layout:
        plt.tight_layout()

    _save_figure(fig, out_path / "figure1_known_condition_clouds.png", cfg)
    plt.close(fig)



# ============================================================
# 图2 / 图3：完整边界值 + 误差
# ============================================================

def _plot_boundary_compare_triplet(
    xy,
    true_field,
    pred_field,
    out_file,
    title_true,
    title_pred,
    title_err,
    cmap_field,
    cfg=PLOT_CFG,
):
    """
    通用的三联图：
    - 真值
    - 预测值
    - 误差
    """
    err = pred_field - true_field

    val_min = min(float(true_field.min().cpu()), float(pred_field.min().cpu()))
    val_max = max(float(true_field.max().cpu()), float(pred_field.max().cpu()))
    err_abs = max(float(torch.max(torch.abs(err)).cpu()), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=cfg.fig23_size)

    sc1 = _scatter_field(
        axes[0], xy, true_field,
        title=title_true,
        cmap=cmap_field,
        vmin=val_min,
        vmax=val_max,
        cfg=cfg,
    )
    _create_colorbar(sc1, axes[0], val_min, val_max, cfg)

    sc2 = _scatter_field(
        axes[1], xy, pred_field,
        title=title_pred,
        cmap=cmap_field,
        vmin=val_min,
        vmax=val_max,
        cfg=cfg,
    )
    _create_colorbar(sc2, axes[1], val_min, val_max, cfg)

    sc3 = _scatter_error(
        axes[2], xy, err,
        title=title_err,
        cmap=cfg.cmap_error,
        cfg=cfg,
    )
    _create_colorbar(sc3, axes[2], -err_abs, err_abs, cfg)

    if cfg.use_tight_layout:
        plt.tight_layout()

    _save_figure(fig, out_file, cfg)
    plt.close(fig)


# ============================================================
# 测试指标计算
# ============================================================

def _compute_boundary_metrics(data, T_full_pred, q_full_pred):
    """
    计算边界预测误差指标：
    - 未知部分 MAE
    - 全局 MAE / MSE
    """
    unknown_T_mask = data.mask_q_known > 0.5
    unknown_q_mask = data.mask_T_known > 0.5

    device = T_full_pred.device

    T_unknown_mae = (
        torch.mean(torch.abs(T_full_pred[unknown_T_mask] - data.T_true[unknown_T_mask]))
        if torch.any(unknown_T_mask)
        else torch.tensor(0.0, device=device)
    )
    q_unknown_mae = (
        torch.mean(torch.abs(q_full_pred[unknown_q_mask] - data.q_true[unknown_q_mask]))
        if torch.any(unknown_q_mask)
        else torch.tensor(0.0, device=device)
    )

    T_global_mae = torch.mean(torch.abs(T_full_pred - data.T_true))
    q_global_mae = torch.mean(torch.abs(q_full_pred - data.q_true))

    T_global_mse = torch.mean((T_full_pred - data.T_true) ** 2)
    q_global_mse = torch.mean((q_full_pred - data.q_true) ** 2)

    return {
        "T_unknown_mae": float(T_unknown_mae.detach().cpu()),
        "q_unknown_mae": float(q_unknown_mae.detach().cpu()),
        "T_global_mae": float(T_global_mae.detach().cpu()),
        "q_global_mae": float(q_global_mae.detach().cpu()),
        "T_global_mse": float(T_global_mse.detach().cpu()),
        "q_global_mse": float(q_global_mse.detach().cpu()),
    }


# ============================================================
# 主测试函数
# ============================================================

def run_test(
    dataset_root="dataset_rect_bem_gnn",
    model_dir="artifacts_train",
    out_dir="artifacts_test",
    cfg=PLOT_CFG,
):
    """
    主测试流程：
    1. 读取测试样本
    2. 加载最佳模型
    3. 预测未知边界值
    4. 恢复完整边界场
    5. 画四张图
    6. 保存测试指标
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 读取测试图
    _, _, test_loader = build_loaders(dataset_root, batch_size=1)
    data = next(iter(test_loader)).to(device)

    # 加载训练好的最佳模型
    ckpt = torch.load(
        Path(model_dir) / "best_model.pt",
        map_location=device,
        weights_only=False,
    )

    model = DualHeadGCNVAE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 前向推理
    with torch.no_grad():
        output = model(data)

        # 用已知值覆盖，恢复完整边界 T / q
        T_full_pred, q_full_pred = assemble_full_fields(
            output.pred_T_unknown,
            output.pred_q_unknown,
            data.T_known,
            data.q_known,
            data.mask_T_known,
            data.mask_q_known,
        )

    # ---------------- 图1：已知边界条件 ----------------
    _plot_known_condition_clouds(data, out_path, cfg=cfg)

    # ---------------- 图2：Dirichlet 完整边界对比 ----------------
    _plot_boundary_compare_triplet(
        xy=data.centers,
        true_field=data.T_true,
        pred_field=T_full_pred,
        out_file=out_path / "figure2_dirichlet_full_boundary_compare.png",
        title_true=cfg.title_dirichlet_true,
        title_pred=cfg.title_dirichlet_pred,
        title_err=cfg.title_dirichlet_err,
        cmap_field=cfg.cmap_dirichlet,
        cfg=cfg,
    )

    # ---------------- 图3：Neumann 完整边界对比 ----------------
    _plot_boundary_compare_triplet(
        xy=data.centers,
        true_field=data.q_true,
        pred_field=q_full_pred,
        out_file=out_path / "figure3_neumann_full_boundary_compare.png",
        title_true=cfg.title_neumann_true,
        title_pred=cfg.title_neumann_pred,
        title_err=cfg.title_neumann_err,
        cmap_field=cfg.cmap_neumann,
        cfg=cfg,
    )

    # ---------------- 图4：域内值对比 ----------------
    U_pred, U_true, U_err = _plot_interior_compare(
        data.boundary_nodes,
        T_full_pred,
        q_full_pred,
        data.T_true,
        data.q_true,
        out_path / "figure4_interior_field_compare.png",
        cfg=cfg,
    )

    # ---------------- 保存指标 ----------------
    metrics = _compute_boundary_metrics(data, T_full_pred, q_full_pred)
    metrics["interior_mae"] = float(np.mean(np.abs(U_pred - U_true)))
    metrics["interior_mse"] = float(np.mean(U_err ** 2))

    if getattr(output, "mu", None) is not None:
        metrics["latent_mu_mean"] = float(output.mu.mean().detach().cpu())
        metrics["latent_mu_std"] = float(output.mu.std().detach().cpu())

    if getattr(output, "logvar", None) is not None:
        metrics["latent_logvar_mean"] = float(output.logvar.mean().detach().cpu())
        metrics["latent_logvar_std"] = float(output.logvar.std().detach().cpu())

    with open(out_path / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 控制台输出
    print("Saved all results to:", out_path.resolve())
    print("Generated figures:")
    print("  1) figure1_known_condition_clouds.png / .svg")
    print("  2) figure2_dirichlet_full_boundary_compare.png / .svg")
    print("  3) figure3_neumann_full_boundary_compare.png / .svg")
    print("  4) figure4_interior_field_compare.png / .svg")
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    run_test()
