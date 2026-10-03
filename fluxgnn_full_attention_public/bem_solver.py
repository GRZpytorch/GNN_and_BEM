from __future__ import annotations

"""Numerical boundary-element operators and field evaluation for the 2D Poisson/Laplace problems."""


from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


_INV_2PI = 1.0 / (2.0 * np.pi)
_EPS = 1e-14


# =========================================================
# Validation utilities
# =========================================================

def _as_points(
    value: np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape [N, 2], "
            f"but received {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise FloatingPointError(
            f"{name} contains NaN or infinite values"
        )

    return array


def _as_vector(
    value: np.ndarray,
    name: str,
    expected_length: int | None = None,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    ).reshape(-1)

    if expected_length is not None:
        if array.shape[0] != expected_length:
            raise ValueError(
                f"{name} contains {array.shape[0]} values, "
                f"but {expected_length} values are required"
            )

    if not np.all(np.isfinite(array)):
        raise FloatingPointError(
            f"{name} contains NaN or infinite values"
        )

    return array


def _validate_domain_quadrature(
    domain_quad: Dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if "points" not in domain_quad:
        raise KeyError(
            "domain_quad does not contain 'points'"
        )

    if "weights" not in domain_quad:
        raise KeyError(
            "domain_quad does not contain 'weights'"
        )

    quad_pts = _as_points(
        domain_quad["points"],
        "domain_quad['points']",
    )

    quad_w = _as_vector(
        domain_quad["weights"],
        "domain_quad['weights']",
        expected_length=quad_pts.shape[0],
    )

    if np.any(quad_w < 0.0):
        raise ValueError(
            "Domain quadrature weights must be non-negative"
        )

    if not np.any(quad_w > 0.0):
        raise ValueError(
            "At least one domain quadrature weight must be positive"
        )

    return quad_pts, quad_w


# =========================================================
# Fundamental solution
# =========================================================

def fundamental_solution(
    eval_pts: np.ndarray,
    src_pts: np.ndarray,
) -> np.ndarray:
    """
    Fundamental solution of the two-dimensional Laplace operator:

        G(x, y) = -(1 / (2*pi)) log(|x-y|).

    With this convention:

        -Delta_y G(x, y) = delta(x-y).

    Parameters
    ----------
    eval_pts:
        Evaluation points with shape [M, 2].

    src_pts:
        Source points with shape [N, 2].

    Returns
    -------
    np.ndarray
        Fundamental-solution matrix with shape [M, N].
    """
    eval_pts = _as_points(
        eval_pts,
        "eval_pts",
    )

    src_pts = _as_points(
        src_pts,
        "src_pts",
    )

    diff = (
        eval_pts[:, None, :]
        - src_pts[None, :, :]
    )

    radius = np.linalg.norm(
        diff,
        axis=2,
    )

    radius = np.maximum(
        radius,
        _EPS,
    )

    return (
        -_INV_2PI
        * np.log(radius)
    )


def dG_dn_source(
    eval_pts: np.ndarray,
    src_pts: np.ndarray,
    src_normals: np.ndarray,
) -> np.ndarray:
    """
    Source-normal derivative of the two-dimensional fundamental solution:

        dG(x, y) / dn_y.

    The normal vectors are defined at the source points y.

    Parameters
    ----------
    eval_pts:
        Evaluation points with shape [M, 2].

    src_pts:
        Source points with shape [N, 2].

    src_normals:
        Source-point outward normals with shape [N, 2].

    Returns
    -------
    np.ndarray
        Source-normal derivative matrix with shape [M, N].
    """
    eval_pts = _as_points(
        eval_pts,
        "eval_pts",
    )

    src_pts = _as_points(
        src_pts,
        "src_pts",
    )

    src_normals = _as_points(
        src_normals,
        "src_normals",
    )

    if src_normals.shape[0] != src_pts.shape[0]:
        raise ValueError(
            "src_normals and src_pts must contain "
            "the same number of points"
        )

    diff = (
        src_pts[None, :, :]
        - eval_pts[:, None, :]
    )

    radius_squared = np.sum(
        diff * diff,
        axis=2,
    )

    radius_squared = np.maximum(
        radius_squared,
        _EPS**2,
    )

    projection = np.sum(
        diff
        * src_normals[None, :, :],
        axis=2,
    )

    return (
        -_INV_2PI
        * projection
        / radius_squared
    )


# =========================================================
# Boundary-panel construction
# =========================================================

def _rotate_ccw(
    vectors: np.ndarray,
) -> np.ndarray:
    vectors = _as_points(
        vectors,
        "vectors",
    )

    return np.stack(
        [
            -vectors[:, 1],
            vectors[:, 0],
        ],
        axis=1,
    )


def _normalize(
    vectors: np.ndarray,
) -> np.ndarray:
    vectors = _as_points(
        vectors,
        "vectors",
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= _EPS):
        indices = np.where(
            norms.reshape(-1) <= _EPS
        )[0]

        raise ValueError(
            "Zero-length normal vectors were found at indices "
            f"{indices.tolist()}"
        )

    return vectors / norms


def _build_local_panels(
    boundary_coords: np.ndarray,
    boundary_normals: np.ndarray,
    boundary_ds: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Construct one local straight panel for each boundary collocation point.

    Input semantics
    ---------------
    boundary_coords[i]:
        Center of boundary panel i.

    boundary_ds[i]:
        Length of boundary panel i.

    boundary_normals[i]:
        Outward unit normal of boundary panel i.

    The panel tangent is obtained by a counterclockwise rotation of the
    outward normal.
    """
    coords = _as_points(
        boundary_coords,
        "boundary_coords",
    )

    normals = _as_points(
        boundary_normals,
        "boundary_normals",
    )

    if normals.shape[0] != coords.shape[0]:
        raise ValueError(
            "boundary_normals and boundary_coords must "
            "contain the same number of entries"
        )

    lengths = _as_vector(
        boundary_ds,
        "boundary_ds",
        expected_length=coords.shape[0],
    )

    if np.any(lengths <= 0.0):
        indices = np.where(
            lengths <= 0.0
        )[0]

        raise ValueError(
            "Boundary-panel lengths must be positive. "
            f"Invalid indices: {indices.tolist()}"
        )

    normalized_normals = _normalize(
        normals
    )

    tangents = _rotate_ccw(
        normalized_normals
    )

    panel_start = (
        coords
        - 0.5
        * lengths[:, None]
        * tangents
    )

    panel_end = (
        coords
        + 0.5
        * lengths[:, None]
        * tangents
    )

    return (
        panel_start,
        panel_end,
        coords.copy(),
        lengths,
        normalized_normals,
        tangents,
    )


# =========================================================
# Boundary-panel quadrature
# =========================================================

def _gauss_legendre_1d(
    nq: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(nq, int):
        raise TypeError(
            "nq must be an integer"
        )

    if nq <= 0:
        raise ValueError(
            "nq must be greater than zero"
        )

    xi, weights = (
        np.polynomial.legendre.leggauss(nq)
    )

    return (
        xi.astype(np.float64),
        weights.astype(np.float64),
    )


def _panel_quadrature_points(
    panel_start: np.ndarray,
    panel_end: np.ndarray,
    nq: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Gauss-Legendre quadrature points and integration weights
    for each straight boundary panel.
    """
    panel_start = _as_points(
        panel_start,
        "panel_start",
    )

    panel_end = _as_points(
        panel_end,
        "panel_end",
    )

    if panel_start.shape != panel_end.shape:
        raise ValueError(
            "panel_start and panel_end must have the same shape"
        )

    xi, wi = _gauss_legendre_1d(
        nq
    )

    tangent_vectors = (
        panel_end
        - panel_start
    )

    lengths = np.linalg.norm(
        tangent_vectors,
        axis=1,
    )

    if np.any(lengths <= _EPS):
        raise ValueError(
            "At least one boundary panel has zero length"
        )

    midpoints = (
        0.5
        * (
            panel_start
            + panel_end
        )
    )

    half_vectors = (
        0.5
        * tangent_vectors
    )

    points = (
        midpoints[:, None, :]
        + xi[None, :, None]
        * half_vectors[:, None, :]
    )

    weights = (
        0.5
        * lengths[:, None]
        * wi[None, :]
    )

    return points, weights


def _integrate_G_over_panels(
    eval_pts: np.ndarray,
    panel_start: np.ndarray,
    panel_end: np.ndarray,
    nq: int = 16,
) -> np.ndarray:
    """
    Compute

        G_ij = integral_{Gamma_j} G(x_i, y) dGamma_y.
    """
    eval_pts = _as_points(
        eval_pts,
        "eval_pts",
    )

    quadrature_points, quadrature_weights = (
        _panel_quadrature_points(
            panel_start,
            panel_end,
            nq=nq,
        )
    )

    num_eval = eval_pts.shape[0]
    num_panels = quadrature_points.shape[0]

    source_points = quadrature_points.reshape(
        num_panels * nq,
        2,
    )

    kernel = fundamental_solution(
        eval_pts,
        source_points,
    ).reshape(
        num_eval,
        num_panels,
        nq,
    )

    return np.sum(
        kernel
        * quadrature_weights[None, :, :],
        axis=2,
    )


def _integrate_dGdn_over_panels(
    eval_pts: np.ndarray,
    panel_start: np.ndarray,
    panel_end: np.ndarray,
    panel_normals: np.ndarray,
    nq: int = 16,
) -> np.ndarray:
    """
    Compute

        H_ij = integral_{Gamma_j}
               dG(x_i, y)/dn_y dGamma_y.
    """
    eval_pts = _as_points(
        eval_pts,
        "eval_pts",
    )

    panel_normals = _as_points(
        panel_normals,
        "panel_normals",
    )

    quadrature_points, quadrature_weights = (
        _panel_quadrature_points(
            panel_start,
            panel_end,
            nq=nq,
        )
    )

    num_panels = quadrature_points.shape[0]
    num_eval = eval_pts.shape[0]

    if panel_normals.shape[0] != num_panels:
        raise ValueError(
            "panel_normals and boundary panels must "
            "have the same length"
        )

    source_points = quadrature_points.reshape(
        num_panels * nq,
        2,
    )

    source_normals = np.repeat(
        panel_normals,
        nq,
        axis=0,
    )

    kernel = dG_dn_source(
        eval_pts,
        source_points,
        source_normals,
    ).reshape(
        num_eval,
        num_panels,
        nq,
    )

    return np.sum(
        kernel
        * quadrature_weights[None, :, :],
        axis=2,
    )


def _panel_self_G(
    lengths: np.ndarray,
) -> np.ndarray:
    """
    Analytical single-layer self-panel integral for midpoint collocation:

        integral_{-L/2}^{L/2}
        -(1/(2*pi)) log|s| ds

        = -(L/(2*pi)) [log(L/2) - 1].
    """
    lengths = _as_vector(
        lengths,
        "lengths",
    )

    if np.any(lengths <= 0.0):
        raise ValueError(
            "Panel lengths must be positive"
        )

    return (
        -_INV_2PI
        * lengths
        * (
            np.log(lengths / 2.0)
            - 1.0
        )
    )


# =========================================================
# Domain integral
# =========================================================

def _domain_self_kernel_average(
    quadrature_weights: np.ndarray,
) -> np.ndarray:
    """
    Approximate the cell-averaged logarithmic kernel at a coincident
    evaluation and quadrature point.

    Each quadrature cell is approximated by a disk with equal area:

        pi * a^2 = w.

    The mean value of G over this disk is

        -(1/(2*pi)) [log(a) - 1/2].

    This correction avoids replacing log(0) by log(EPS), which can
    produce an artificially large source contribution.
    """
    quadrature_weights = _as_vector(
        quadrature_weights,
        "quadrature_weights",
    )

    positive_weights = np.maximum(
        quadrature_weights,
        _EPS,
    )

    equivalent_radius = np.sqrt(
        positive_weights / np.pi
    )

    return (
        -_INV_2PI
        * (
            np.log(equivalent_radius)
            - 0.5
        )
    )


def _evaluate_domain_term(
    eval_pts: np.ndarray,
    quad_pts: np.ndarray,
    quad_w: np.ndarray,
    source_values: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    Evaluate

        integral_Omega G(x, y) f(y) dOmega_y

    using discrete domain quadrature.

    A local cell-average correction is applied when an evaluation point
    coincides with a quadrature point.
    """
    eval_pts = _as_points(
        eval_pts,
        "eval_pts",
    )

    quad_pts = _as_points(
        quad_pts,
        "quad_pts",
    )

    quad_w = _as_vector(
        quad_w,
        "quad_w",
        expected_length=quad_pts.shape[0],
    )

    source_values = _as_vector(
        source_values,
        "source_values",
        expected_length=quad_pts.shape[0],
    )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    weighted_source = (
        source_values
        * quad_w
    )

    output = np.empty(
        eval_pts.shape[0],
        dtype=np.float64,
    )

    self_kernel_average = (
        _domain_self_kernel_average(
            quad_w
        )
    )

    coordinate_scale = max(
        float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            eval_pts.reshape(-1),
                            quad_pts.reshape(-1),
                        ]
                    )
                )
            )
        ),
        1.0,
    )

    coincidence_tolerance = max(
        1e-12 * coordinate_scale,
        _EPS,
    )

    tolerance_squared = (
        coincidence_tolerance**2
    )

    for start in range(
        0,
        eval_pts.shape[0],
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            eval_pts.shape[0],
        )

        eval_chunk = eval_pts[start:stop]

        difference = (
            eval_chunk[:, None, :]
            - quad_pts[None, :, :]
        )

        radius_squared = np.sum(
            difference * difference,
            axis=2,
        )

        kernel = fundamental_solution(
            eval_chunk,
            quad_pts,
        )

        coincident = (
            radius_squared
            <= tolerance_squared
        )

        if np.any(coincident):
            row_indices, column_indices = (
                np.nonzero(coincident)
            )

            kernel[
                row_indices,
                column_indices,
            ] = self_kernel_average[
                column_indices
            ]

        output[start:stop] = (
            kernel
            @ weighted_source
        )

    return output


# =========================================================
# BEM cache
# =========================================================

@dataclass
class BEMCache:
    boundary_coords: np.ndarray
    boundary_normals: np.ndarray
    boundary_ds: np.ndarray

    quad_pts: np.ndarray
    quad_w: np.ndarray

    G_bb: np.ndarray
    H_bb: np.ndarray
    A_base: np.ndarray
    G_bq: np.ndarray

    panel_start: np.ndarray
    panel_end: np.ndarray
    panel_mid: np.ndarray
    panel_len: np.ndarray
    panel_normals: np.ndarray

    nq_boundary: int = 16

    @classmethod
    def build(
        cls,
        boundary_coords: np.ndarray,
        boundary_normals: np.ndarray,
        boundary_ds: np.ndarray,
        domain_quad: Dict[str, np.ndarray],
        nq_boundary: int = 16,
    ) -> "BEMCache":
        """
        Assemble and cache the boundary-element matrices.

        Each boundary_coords[i] is interpreted as the midpoint of one
        straight constant boundary element.
        """
        if not isinstance(nq_boundary, int):
            raise TypeError(
                "nq_boundary must be an integer"
            )

        if nq_boundary <= 0:
            raise ValueError(
                "nq_boundary must be greater than zero"
            )

        coords = _as_points(
            boundary_coords,
            "boundary_coords",
        )

        normals = _as_points(
            boundary_normals,
            "boundary_normals",
        )

        if normals.shape[0] != coords.shape[0]:
            raise ValueError(
                "boundary_normals and boundary_coords must "
                "contain the same number of entries"
            )

        ds = _as_vector(
            boundary_ds,
            "boundary_ds",
            expected_length=coords.shape[0],
        )

        quad_pts, quad_w = (
            _validate_domain_quadrature(
                domain_quad
            )
        )

        (
            panel_start,
            panel_end,
            panel_mid,
            panel_len,
            panel_normals,
            _,
        ) = _build_local_panels(
            boundary_coords=coords,
            boundary_normals=normals,
            boundary_ds=ds,
        )

        G_bb = _integrate_G_over_panels(
            eval_pts=panel_mid,
            panel_start=panel_start,
            panel_end=panel_end,
            nq=nq_boundary,
        )

        H_bb = _integrate_dGdn_over_panels(
            eval_pts=panel_mid,
            panel_start=panel_start,
            panel_end=panel_end,
            panel_normals=panel_normals,
            nq=nq_boundary,
        )

        diagonal_indices = np.arange(
            coords.shape[0]
        )

        # Analytical single-layer self-panel term.
        G_bb[
            diagonal_indices,
            diagonal_indices,
        ] = _panel_self_G(
            panel_len
        )

        # For a straight constant element and midpoint collocation,
        # the principal-value self contribution of dG/dn_y is zero.
        H_bb[
            diagonal_indices,
            diagonal_indices,
        ] = 0.0

        # The unknown flux system is:
        #
        #     G_bb q = 0.5 u + H_bb u - F.
        A_base = G_bb.copy()

        # Boundary-to-domain-quadrature kernel.
        # Boundary and interior quadrature points should not coincide.
        G_bq = fundamental_solution(
            panel_mid,
            quad_pts,
        )

        return cls(
            boundary_coords=coords,
            boundary_normals=panel_normals,
            boundary_ds=ds,

            quad_pts=quad_pts,
            quad_w=quad_w,

            G_bb=G_bb,
            H_bb=H_bb,
            A_base=A_base,
            G_bq=G_bq,

            panel_start=panel_start,
            panel_end=panel_end,
            panel_mid=panel_mid,
            panel_len=panel_len,
            panel_normals=panel_normals,

            nq_boundary=nq_boundary,
        )


# =========================================================
# Boundary flux solution
# =========================================================

def solve_flux_collocation_cached(
    cache: BEMCache,
    u_bc: np.ndarray,
    f_fn: Callable[[np.ndarray], np.ndarray],
    regularization: float = 1e-10,
) -> np.ndarray:
    """
    Solve the complete Dirichlet-to-Neumann problem:

        known u on the complete boundary -> unknown q.

    Discrete equation:

        G_bb q
        =
        0.5 u
        + H_bb u
        - F.

    This function is not a general mixed-boundary solver.
    """
    if regularization < 0.0:
        raise ValueError(
            "regularization must be non-negative"
        )

    num_boundary_nodes = (
        cache.boundary_coords.shape[0]
    )

    u_bc = _as_vector(
        u_bc,
        "u_bc",
        expected_length=num_boundary_nodes,
    )

    source_values = _as_vector(
        f_fn(cache.quad_pts),
        "f_fn(cache.quad_pts)",
        expected_length=cache.quad_pts.shape[0],
    )

    source_term = (
        cache.G_bq
        @ (
            source_values
            * cache.quad_w
        )
    )

    rhs = (
        0.5 * u_bc
        + cache.H_bb @ u_bc
        - source_term
    )

    system_matrix = (
        cache.A_base.copy()
    )

    if regularization > 0.0:
        diagonal_indices = np.diag_indices_from(
            system_matrix
        )

        system_matrix[
            diagonal_indices
        ] += regularization

    try:
        flux = np.linalg.solve(
            system_matrix,
            rhs,
        )
    except np.linalg.LinAlgError as error:
        condition_number = np.linalg.cond(
            system_matrix
        )

        raise np.linalg.LinAlgError(
            "Failed to solve the boundary flux system. "
            f"Estimated condition number: {condition_number:.6e}"
        ) from error

    if not np.all(np.isfinite(flux)):
        raise FloatingPointError(
            "The computed boundary flux contains "
            "NaN or infinite values"
        )

    return flux


def solve_flux_collocation_fast(
    cache: BEMCache,
    u_bc: np.ndarray,
    f_fn: Callable[[np.ndarray], np.ndarray],
    regularization: float = 1e-10,
) -> np.ndarray:
    """
    Compatibility alias of solve_flux_collocation_cached().
    """
    return solve_flux_collocation_cached(
        cache=cache,
        u_bc=u_bc,
        f_fn=f_fn,
        regularization=regularization,
    )


def solve_flux_collocation(
    boundary_coords: np.ndarray,
    boundary_normals: np.ndarray,
    boundary_ds: np.ndarray,
    u_bc: np.ndarray,
    f_fn: Callable[[np.ndarray], np.ndarray],
    domain_quad: Dict[str, np.ndarray],
    regularization: float = 1e-10,
) -> np.ndarray:
    """
    Assemble the cache and solve the complete Dirichlet-to-Neumann problem.
    """
    cache = BEMCache.build(
        boundary_coords=boundary_coords,
        boundary_normals=boundary_normals,
        boundary_ds=boundary_ds,
        domain_quad=domain_quad,
    )

    return solve_flux_collocation_cached(
        cache=cache,
        u_bc=u_bc,
        f_fn=f_fn,
        regularization=regularization,
    )


# =========================================================
# Interior field evaluation
# =========================================================

def evaluate_field(
    eval_pts: np.ndarray,
    boundary_coords: np.ndarray,
    boundary_normals: np.ndarray,
    boundary_ds: np.ndarray,
    u_bc: np.ndarray,
    q_bc: np.ndarray,
    f_fn: Callable[[np.ndarray], np.ndarray],
    domain_quad: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Assemble a BEM cache and evaluate the physical field at interior points.

    This function preserves the original public interface. When a cache
    is already available, evaluate_field_fast() should be used to avoid
    rebuilding the boundary matrices.
    """
    cache = BEMCache.build(
        boundary_coords=boundary_coords,
        boundary_normals=boundary_normals,
        boundary_ds=boundary_ds,
        domain_quad=domain_quad,
    )

    return evaluate_field_fast(
        eval_pts=eval_pts,
        cache=cache,
        u_bc=u_bc,
        q_bc=q_bc,
        f_fn=f_fn,
    )


def evaluate_field_fast(
    eval_pts: np.ndarray,
    cache: BEMCache,
    u_bc: np.ndarray,
    q_bc: np.ndarray,
    f_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Evaluate the interior field using an existing BEM cache.

    Representation formula:

        u(x)
        =
        integral_Gamma G(x, y) q(y) dGamma_y
        - integral_Gamma dG(x, y)/dn_y u(y) dGamma_y
        + integral_Omega G(x, y) f(y) dOmega_y.
    """
    eval_pts = _as_points(
        eval_pts,
        "eval_pts",
    )

    num_boundary_nodes = (
        cache.boundary_coords.shape[0]
    )

    u_bc = _as_vector(
        u_bc,
        "u_bc",
        expected_length=num_boundary_nodes,
    )

    q_bc = _as_vector(
        q_bc,
        "q_bc",
        expected_length=num_boundary_nodes,
    )

    single_layer_matrix = (
        _integrate_G_over_panels(
            eval_pts=eval_pts,
            panel_start=cache.panel_start,
            panel_end=cache.panel_end,
            nq=cache.nq_boundary,
        )
    )

    double_layer_matrix = (
        _integrate_dGdn_over_panels(
            eval_pts=eval_pts,
            panel_start=cache.panel_start,
            panel_end=cache.panel_end,
            panel_normals=cache.panel_normals,
            nq=cache.nq_boundary,
        )
    )

    boundary_term = (
        single_layer_matrix @ q_bc
        - double_layer_matrix @ u_bc
    )

    source_values = _as_vector(
        f_fn(cache.quad_pts),
        "f_fn(cache.quad_pts)",
        expected_length=cache.quad_pts.shape[0],
    )

    domain_term = _evaluate_domain_term(
        eval_pts=eval_pts,
        quad_pts=cache.quad_pts,
        quad_w=cache.quad_w,
        source_values=source_values,
    )

    field = (
        boundary_term
        + domain_term
    )

    if not np.all(np.isfinite(field)):
        raise FloatingPointError(
            "The evaluated interior field contains "
            "NaN or infinite values"
        )

    return field
