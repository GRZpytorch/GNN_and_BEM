"""Optional standalone visualization example for a smooth closed boundary."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path

def smooth_closed_curve(points, n_interp=500, smooth=0.0):
    points = np.asarray(points)
    x, y = (points[:, 0], points[:, 1])
    tck, _ = splprep([x, y], s=smooth, per=True)
    u_new = np.linspace(0, 1, n_interp)
    x_new, y_new = splev(u_new, tck)
    curve = np.column_stack([x_new, y_new])
    return curve

def make_segment_colors(curve, top_color='#b71c1c', bottom_color='#0d47a1'):
    y = curve[:, 1]
    y_min, y_max = (y.min(), y.max())
    y_norm = (y - y_min) / (y_max - y_min + 1e-12)
    cmap = LinearSegmentedColormap.from_list('scientific_red_blue', [bottom_color, '#5c88c4', '#f3f3f3', '#d97b7b', top_color])
    return cmap(y_norm)

def create_scientific_gradient_background(ax, curve, extent_pad=0.8, top_color='#d32f2f', bottom_color='#1976d2', alpha=0.72, resolution=700):
    x = curve[:, 0]
    y = curve[:, 1]
    xmin, xmax = (x.min() - extent_pad, x.max() + extent_pad)
    ymin, ymax = (y.min() - extent_pad, y.max() + extent_pad)
    xx = np.linspace(xmin, xmax, resolution)
    yy = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(xx, yy)
    Yn = (Y - ymin) / (ymax - ymin + 1e-12)
    cmap = LinearSegmentedColormap.from_list('cloud_fill_stronger', [bottom_color, '#6ea6de', '#eef3f8', '#f3b0aa', top_color])
    rgba = cmap(Yn)
    center_weight = 1.0 - 0.18 * np.abs(Yn - 0.5) / 0.5
    rgba[..., -1] = alpha * center_weight
    path = Path(curve)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = path.contains_points(pts).reshape(X.shape)
    rgba[..., -1] *= mask.astype(float)
    ax.imshow(rgba, origin='lower', extent=[xmin, xmax, ymin, ymax], interpolation='bicubic', aspect='auto', zorder=1)

def plot_cloud_shape(control_points, point_colors=None, top_color='#b71c1c', bottom_color='#0d47a1', curve_width=4.2, point_size=95, smooth=0.15, show_labels=True):
    control_points = np.asarray(control_points)
    assert control_points.shape == (12, 2), 'Expected 12 two-dimensional boundary control points with shape (12, 2).'
    if point_colors is None:
        y_mid = np.mean(control_points[:, 1])
        point_colors = [top_color if p[1] >= y_mid else bottom_color for p in control_points]
    curve = smooth_closed_curve(control_points, n_interp=800, smooth=smooth)
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    create_scientific_gradient_background(ax, curve, top_color='#d32f2f', bottom_color='#1976d2', alpha=0.72, resolution=900)
    seg_points = curve.reshape(-1, 1, 2)
    segments = np.concatenate([seg_points[:-1], seg_points[1:]], axis=1)
    seg_colors = make_segment_colors(curve, top_color=top_color, bottom_color=bottom_color)
    ax.plot(curve[:, 0], curve[:, 1], color='white', lw=6.0, alpha=0.9, solid_joinstyle='round', solid_capstyle='round', zorder=2)
    ax.plot(curve[:, 0], curve[:, 1], color='#222222', lw=4.8, alpha=0.95, solid_joinstyle='round', solid_capstyle='round', zorder=3)
    lc = LineCollection(segments, colors=seg_colors[:-1], linewidths=curve_width, capstyle='round', joinstyle='round', zorder=4)
    ax.add_collection(lc)
    ax.scatter(control_points[:, 0], control_points[:, 1], s=point_size, c=point_colors, edgecolors='white', linewidths=1.6, zorder=5)
    ax.scatter(control_points[:, 0], control_points[:, 1], s=point_size, facecolors='none', edgecolors='#1a1a1a', linewidths=0.9, zorder=6)
    if show_labels:
        for i, (x, y) in enumerate(control_points):
            ax.text(x + 0.06, y + 0.06, f'P{i + 1}', fontsize=9, color='#111111', weight='semibold', zorder=7)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Smooth Cloud Shape with 12 Boundary Control Points', fontsize=13, pad=10)
    ax.grid(True, linestyle='--', linewidth=0.55, alpha=0.22, color='0.35')
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color('#333333')
        spine.set_alpha(0.95)
    ax.tick_params(direction='in', length=5, width=0.9, colors='#222222')
    margin = 0.8
    ax.set_xlim(control_points[:, 0].min() - margin, control_points[:, 0].max() + margin)
    ax.set_ylim(control_points[:, 1].min() - margin, control_points[:, 1].max() + margin)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    control_points = np.array([[-2.8, 0.2], [-2.2, 1.1], [-1.2, 1.9], [0.0, 2.2], [1.3, 1.8], [2.3, 1.0], [2.9, 0.1], [2.4, -0.8], [1.4, -1.5], [0.1, -1.8], [-1.3, -1.4], [-2.4, -0.7]])
    y_mid = np.mean(control_points[:, 1])
    point_colors = ['#c62828' if p[1] >= y_mid else '#1565c0' for p in control_points]
    plot_cloud_shape(control_points=control_points, point_colors=point_colors, top_color='#b71c1c', bottom_color='#0d47a1', curve_width=4.2, point_size=95, smooth=0.18, show_labels=True)
