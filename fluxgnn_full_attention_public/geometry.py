from __future__ import annotations

"""Boundary discretization, graph construction, and interior quadrature utilities."""


import math
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np


GeometryType = Literal["disk", "square"]

_EPS = 1e-12


# =========================================================
# Data structures
# =========================================================

@dataclass
class BoundaryMesh:
    """
    Discretized boundary representation.

    Attributes
    ----------
    coords:
        Boundary-element collocation points with shape [N, 2].

    normals:
        Unit outward normal vectors with shape [N, 2].

    ds:
        Boundary-element lengths with shape [N].

    edge_index:
        Directed graph edges with shape [2, E].

    geometry_type:
        Geometry identifier: "disk" or "square".
    """

    coords: np.ndarray
    normals: np.ndarray
    ds: np.ndarray
    edge_index: np.ndarray
    geometry_type: str


@dataclass
class DomainGrid:
    """
    Cartesian grid used for evaluating and visualizing the interior field.

    Attributes
    ----------
    points:
        All Cartesian grid points with shape [M, 2].

    mask:
        Boolean mask identifying strictly interior points.

    shape:
        Two-dimensional grid shape.

    geometry_type:
        Geometry identifier.
    """

    points: np.ndarray
    mask: np.ndarray
    shape: Tuple[int, int]
    geometry_type: str


# =========================================================
# Validation utilities
# =========================================================

def _validate_positive_integer(
    value: int,
    name: str,
    minimum: int = 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an integer, "
            f"but received {type(value).__name__}"
        )

    value = int(value)

    if value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}, "
            f"but received {value}"
        )

    return value


def _validate_geometry_type(
    geometry_type: str,
) -> GeometryType:
    if geometry_type not in {"disk", "square"}:
        raise ValueError(
            f"Unsupported geometry_type={geometry_type!r}. "
            "Expected 'disk' or 'square'."
        )

    return geometry_type  # type: ignore[return-value]


def _normalize_rows(
    vectors: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(
        vectors,
        dtype=np.float64,
    )

    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError(
            "vectors must have shape [N, 2]"
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= _EPS):
        invalid = np.where(
            norms.reshape(-1) <= _EPS
        )[0]

        raise ValueError(
            "Zero-length vectors were found at indices "
            f"{invalid.tolist()}"
        )

    return vectors / norms


# =========================================================
# Graph construction
# =========================================================

def ring_edges(
    n: int,
) -> np.ndarray:
    """
    Construct a bidirectional ring graph.

    This function is retained for compatibility and for experiments
    based on local boundary adjacency. The boundary-generation functions
    below use fully_connected_edges() by default.
    """
    n = _validate_positive_integer(
        n,
        "n",
        minimum=2,
    )

    source = np.arange(
        n,
        dtype=np.int64,
    )

    target = (
        source + 1
    ) % n

    return np.stack(
        [
            np.concatenate(
                [source, target]
            ),
            np.concatenate(
                [target, source]
            ),
        ],
        axis=0,
    )


def fully_connected_edges(
    n: int,
    include_self: bool = False,
) -> np.ndarray:
    """
    Construct a directed fully connected graph.

    For include_self=False, every ordered node pair i -> j with i != j
    is included. The number of directed edges is N(N-1).

    Self-edges are omitted by default because the attention model already
    preserves node-specific information through its residual connection.
    """
    n = _validate_positive_integer(
        n,
        "n",
        minimum=1,
    )

    nodes = np.arange(
        n,
        dtype=np.int64,
    )

    source = np.repeat(
        nodes,
        n,
    )

    target = np.tile(
        nodes,
        n,
    )

    if not include_self:
        keep = source != target
        source = source[keep]
        target = target[keep]

    return np.stack(
        [source, target],
        axis=0,
    )


# =========================================================
# Boundary generation
# =========================================================

def make_unit_circle_boundary(
    n_boundary: int,
) -> BoundaryMesh:
    """
    Discretize the unit-circle boundary.

    Each node represents the center of a curved boundary segment.
    The BEM implementation approximates this segment locally using a
    straight tangent panel with the same arc length.

    A fully connected directed graph is generated to represent the
    nonlocal interaction among all boundary elements.
    """
    n_boundary = _validate_positive_integer(
        n_boundary,
        "n_boundary",
        minimum=3,
    )

    angular_step = (
        2.0
        * math.pi
        / n_boundary
    )

    # Use segment-center angles rather than segment-end angles.
    theta = (
        np.arange(
            n_boundary,
            dtype=np.float64,
        )
        + 0.5
    ) * angular_step

    coords_64 = np.stack(
        [
            np.cos(theta),
            np.sin(theta),
        ],
        axis=1,
    )

    normals_64 = _normalize_rows(
        coords_64
    )

    # Arc length associated with each boundary node.
    ds_64 = np.full(
        (n_boundary,),
        angular_step,
        dtype=np.float64,
    )

    edge_index = fully_connected_edges(
        n_boundary,
        include_self=False,
    )

    return BoundaryMesh(
        coords=coords_64.astype(np.float32),
        normals=normals_64.astype(np.float32),
        ds=ds_64.astype(np.float32),
        edge_index=edge_index,
        geometry_type="disk",
    )


def make_unit_square_boundary(
    n_per_side: int,
) -> BoundaryMesh:
    """
    Discretize the boundary of the unit square.

    The boundary nodes are constant-element collocation points located
    at the centers of the boundary panels. This definition is consistent
    with BEMCache.build(), which interprets boundary_coords[i] as the
    center of panel i and boundary_ds[i] as its length.

    The nodes are ordered counterclockwise:

        bottom -> right -> top -> left.
    """
    n_per_side = _validate_positive_integer(
        n_per_side,
        "n_per_side",
        minimum=2,
    )

    panel_length = (
        1.0
        / n_per_side
    )

    # Midpoint coordinate along each side:
    #
    #     h/2, 3h/2, ..., 1-h/2.
    side_parameter = (
        np.arange(
            n_per_side,
            dtype=np.float64,
        )
        + 0.5
    ) * panel_length

    zeros = np.zeros_like(
        side_parameter
    )

    ones = np.ones_like(
        side_parameter
    )

    # Counterclockwise boundary ordering.
    bottom = np.stack(
        [
            side_parameter,
            zeros,
        ],
        axis=1,
    )

    right = np.stack(
        [
            ones,
            side_parameter,
        ],
        axis=1,
    )

    top = np.stack(
        [
            1.0 - side_parameter,
            ones,
        ],
        axis=1,
    )

    left = np.stack(
        [
            zeros,
            1.0 - side_parameter,
        ],
        axis=1,
    )

    coords_64 = np.concatenate(
        [
            bottom,
            right,
            top,
            left,
        ],
        axis=0,
    )

    bottom_normals = np.tile(
        np.array(
            [[0.0, -1.0]],
            dtype=np.float64,
        ),
        (n_per_side, 1),
    )

    right_normals = np.tile(
        np.array(
            [[1.0, 0.0]],
            dtype=np.float64,
        ),
        (n_per_side, 1),
    )

    top_normals = np.tile(
        np.array(
            [[0.0, 1.0]],
            dtype=np.float64,
        ),
        (n_per_side, 1),
    )

    left_normals = np.tile(
        np.array(
            [[-1.0, 0.0]],
            dtype=np.float64,
        ),
        (n_per_side, 1),
    )

    normals_64 = np.concatenate(
        [
            bottom_normals,
            right_normals,
            top_normals,
            left_normals,
        ],
        axis=0,
    )

    num_boundary_nodes = int(
        coords_64.shape[0]
    )

    ds_64 = np.full(
        (num_boundary_nodes,),
        panel_length,
        dtype=np.float64,
    )

    edge_index = fully_connected_edges(
        num_boundary_nodes,
        include_self=False,
    )

    return BoundaryMesh(
        coords=coords_64.astype(np.float32),
        normals=normals_64.astype(np.float32),
        ds=ds_64.astype(np.float32),
        edge_index=edge_index,
        geometry_type="square",
    )


# =========================================================
# Interior evaluation grid
# =========================================================

def make_domain_grid(
    geometry_type: str,
    resolution: int = 121,
) -> DomainGrid:
    """
    Construct a Cartesian grid for interior-field evaluation.

    The returned points contain the full Cartesian plotting grid, while
    mask identifies strictly interior points. Boundary points are excluded
    because evaluate_field_fast() uses the interior representation formula
    with coefficient c(x)=1.
    """
    geometry_type = _validate_geometry_type(
        geometry_type
    )

    resolution = _validate_positive_integer(
        resolution,
        "resolution",
        minimum=3,
    )

    if geometry_type == "disk":
        x_values = np.linspace(
            -1.0,
            1.0,
            resolution,
            dtype=np.float64,
        )

        y_values = np.linspace(
            -1.0,
            1.0,
            resolution,
            dtype=np.float64,
        )

        xx, yy = np.meshgrid(
            x_values,
            y_values,
            indexing="xy",
        )

        points_64 = np.stack(
            [
                xx.ravel(),
                yy.ravel(),
            ],
            axis=1,
        )

        radius_squared = (
            points_64[:, 0] ** 2
            + points_64[:, 1] ** 2
        )

        # Strictly interior points only.
        mask = (
            radius_squared
            < (1.0 - _EPS) ** 2
        )

        return DomainGrid(
            points=points_64.astype(np.float32),
            mask=mask.astype(bool),
            shape=xx.shape,
            geometry_type="disk",
        )

    x_values = np.linspace(
        0.0,
        1.0,
        resolution,
        dtype=np.float64,
    )

    y_values = np.linspace(
        0.0,
        1.0,
        resolution,
        dtype=np.float64,
    )

    xx, yy = np.meshgrid(
        x_values,
        y_values,
        indexing="xy",
    )

    points_64 = np.stack(
        [
            xx.ravel(),
            yy.ravel(),
        ],
        axis=1,
    )

    x = points_64[:, 0]
    y = points_64[:, 1]

    # Exclude the square boundary. The BEM interior representation
    # is not valid at boundary collocation points without the c=1/2 term.
    mask = (
        (x > _EPS)
        & (x < 1.0 - _EPS)
        & (y > _EPS)
        & (y < 1.0 - _EPS)
    )

    return DomainGrid(
        points=points_64.astype(np.float32),
        mask=mask.astype(bool),
        shape=xx.shape,
        geometry_type="square",
    )


# =========================================================
# Domain quadrature
# =========================================================

def make_domain_quadrature(
    geometry_type: str,
    resolution: int = 121,
) -> Dict[str, np.ndarray]:
    """
    Construct midpoint quadrature points for the domain source integral.

    The previous implementation placed quadrature points on the same grid
    used for field evaluation and assigned h^2 to every point. For the
    square, this produced a total quadrature weight larger than the domain
    area because both endpoints were included.

    This implementation divides the bounding box into
    (resolution-1) x (resolution-1) cells and places one quadrature point
    at each cell center. Therefore:

    - square quadrature weights sum exactly to 1;
    - quadrature points do not lie on the boundary;
    - quadrature points do not coincide with the evaluation-grid points;
    - logarithmic-kernel coincidence is substantially reduced.
    """
    geometry_type = _validate_geometry_type(
        geometry_type
    )

    resolution = _validate_positive_integer(
        resolution,
        "resolution",
        minimum=2,
    )

    num_cells = (
        resolution - 1
    )

    if geometry_type == "square":
        lower = 0.0
        upper = 1.0
    else:
        lower = -1.0
        upper = 1.0

    cell_size = (
        upper - lower
    ) / num_cells

    cell_centers_1d = (
        lower
        + (
            np.arange(
                num_cells,
                dtype=np.float64,
            )
            + 0.5
        )
        * cell_size
    )

    xx, yy = np.meshgrid(
        cell_centers_1d,
        cell_centers_1d,
        indexing="xy",
    )

    candidate_points = np.stack(
        [
            xx.ravel(),
            yy.ravel(),
        ],
        axis=1,
    )

    if geometry_type == "disk":
        radius_squared = (
            candidate_points[:, 0] ** 2
            + candidate_points[:, 1] ** 2
        )

        inside = (
            radius_squared < 1.0
        )

        quadrature_points = (
            candidate_points[inside]
        )
    else:
        quadrature_points = (
            candidate_points
        )

    quadrature_weights = np.full(
        (quadrature_points.shape[0],),
        cell_size**2,
        dtype=np.float64,
    )

    if quadrature_points.shape[0] == 0:
        raise RuntimeError(
            "Domain quadrature did not generate any points"
        )

    return {
        "points": quadrature_points.astype(
            np.float32
        ),
        "weights": quadrature_weights.astype(
            np.float32
        ),
    }


# =========================================================
# Optional diagnostics
# =========================================================

def boundary_mesh_diagnostics(
    mesh: BoundaryMesh,
) -> Dict[str, float | int | str]:
    """
    Return basic diagnostic information for a boundary mesh.
    """
    coords = np.asarray(
        mesh.coords,
        dtype=np.float64,
    )

    normals = np.asarray(
        mesh.normals,
        dtype=np.float64,
    )

    ds = np.asarray(
        mesh.ds,
        dtype=np.float64,
    ).reshape(-1)

    edge_index = np.asarray(
        mesh.edge_index,
        dtype=np.int64,
    )

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            "mesh.coords must have shape [N, 2]"
        )

    if normals.shape != coords.shape:
        raise ValueError(
            "mesh.normals must have the same shape as mesh.coords"
        )

    if ds.shape[0] != coords.shape[0]:
        raise ValueError(
            "mesh.ds must contain one value per boundary node"
        )

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            "mesh.edge_index must have shape [2, E]"
        )

    normal_norms = np.linalg.norm(
        normals,
        axis=1,
    )

    return {
        "geometry_type": mesh.geometry_type,
        "num_boundary_nodes": int(
            coords.shape[0]
        ),
        "num_directed_edges": int(
            edge_index.shape[1]
        ),
        "total_boundary_measure": float(
            np.sum(ds)
        ),
        "minimum_panel_length": float(
            np.min(ds)
        ),
        "maximum_panel_length": float(
            np.max(ds)
        ),
        "minimum_normal_norm": float(
            np.min(normal_norms)
        ),
        "maximum_normal_norm": float(
            np.max(normal_norms)
        ),
    }


def domain_quadrature_diagnostics(
    geometry_type: str,
    resolution: int = 121,
) -> Dict[str, float | int | str]:
    """
    Return basic diagnostic information for the domain quadrature.
    """
    quadrature = make_domain_quadrature(
        geometry_type,
        resolution=resolution,
    )

    points = np.asarray(
        quadrature["points"],
        dtype=np.float64,
    )

    weights = np.asarray(
        quadrature["weights"],
        dtype=np.float64,
    )

    exact_area = (
        math.pi
        if geometry_type == "disk"
        else 1.0
    )

    numerical_area = float(
        np.sum(weights)
    )

    return {
        "geometry_type": geometry_type,
        "num_quadrature_points": int(
            points.shape[0]
        ),
        "numerical_area": numerical_area,
        "exact_area": exact_area,
        "absolute_area_error": abs(
            numerical_area - exact_area
        ),
        "relative_area_error": abs(
            numerical_area - exact_area
        ) / exact_area,
    }
