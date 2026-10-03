from __future__ import annotations

"""Analytical benchmark problems used for data generation and evaluation."""


import math
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


# =========================================================
# Type definitions
# =========================================================

ScalarField = Callable[
    [np.ndarray],
    np.ndarray,
]

BoundaryFlux = Callable[
    [np.ndarray, np.ndarray],
    np.ndarray,
]


@dataclass
class ProblemSample:
    """
    Analytical problem definition.

    Attributes
    ----------
    case_name:
        Problem identifier.

    geometry_type:
        Geometry identifier used by geometry.py.

    params:
        Global physical parameters stored as a one-dimensional array.

    u_fn:
        Exact scalar field u(x).

    f_fn:
        Source term f(x) corresponding to

            -Delta u = f.

    q_fn:
        Exact outward normal derivative

            q = grad(u) dot n.
    """

    case_name: str
    geometry_type: str
    params: np.ndarray
    u_fn: ScalarField
    f_fn: ScalarField
    q_fn: BoundaryFlux


# =========================================================
# Validation utilities
# =========================================================

def _validate_rng(
    rng: np.random.Generator,
) -> np.random.Generator:
    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError(
            "rng must be an instance of numpy.random.Generator"
        )

    return rng


def _ensure_2d(
    x: np.ndarray,
    name: str = "x",
) -> np.ndarray:
    """
    Convert coordinates to a finite float64 array with shape [N, 2].

    A single coordinate with shape [2] is converted to [1, 2].
    """
    array = np.asarray(
        x,
        dtype=np.float64,
    )

    if array.ndim == 1:
        if array.shape[0] != 2:
            raise ValueError(
                f"{name} must contain two coordinates, "
                f"but received shape {array.shape}"
            )

        array = array[None, :]

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape [N, 2], "
            f"but received {array.shape}"
        )

    if array.shape[0] == 0:
        raise ValueError(
            f"{name} must contain at least one point"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise FloatingPointError(
            f"{name} contains NaN or infinite values"
        )

    return np.ascontiguousarray(
        array,
        dtype=np.float64,
    )


def _prepare_points_and_normals(
    x: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate boundary coordinates and associated normal vectors.
    """
    points = _ensure_2d(
        x,
        name="x",
    )

    normal_vectors = _ensure_2d(
        normals,
        name="normals",
    )

    if normal_vectors.shape[0] != points.shape[0]:
        raise ValueError(
            f"x contains {points.shape[0]} points, while normals "
            f"contains {normal_vectors.shape[0]} vectors"
        )

    normal_norms = np.linalg.norm(
        normal_vectors,
        axis=1,
    )

    if np.any(
        normal_norms <= 1e-14
    ):
        invalid_indices = np.where(
            normal_norms <= 1e-14
        )[0]

        raise ValueError(
            "Zero-length normal vectors were found at indices "
            f"{invalid_indices.tolist()}"
        )

    # Normalize the supplied vectors so q_fn remains valid even if
    # the input normals contain small normalization errors.
    normal_vectors = (
        normal_vectors
        / normal_norms[:, None]
    )

    return points, normal_vectors


def _build_params(
    *values: float,
) -> np.ndarray:
    """
    Construct the stored parameter vector.

    float32 is retained to match the dataset storage convention.
    """
    params = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1)

    if not np.all(
        np.isfinite(params)
    ):
        raise FloatingPointError(
            "Problem parameters contain NaN or infinite values"
        )

    return params


# =========================================================
# Case 1
# =========================================================

def sample_case1(
    rng: np.random.Generator,
) -> ProblemSample:
    """
    Constant-source Poisson problem on the unit disk.

    Exact solution:

        u(x, y) = c/4 * (1 - x^2 - y^2).

    Governing equation:

        -Delta u = c.

    On the unit-circle boundary:

        u = 0,
        q = grad(u) dot n = -c/2.
    """
    rng = _validate_rng(
        rng
    )

    source_amplitude = float(
        rng.uniform(
            0.2,
            2.0,
        )
    )

    params = _build_params(
        source_amplitude,
        0.0,
        0.0,
    )

    def u_fn(
        x: np.ndarray,
    ) -> np.ndarray:
        points = _ensure_2d(
            x,
            name="x",
        )

        radius_squared = np.sum(
            points**2,
            axis=1,
        )

        values = (
            source_amplitude
            / 4.0
        ) * (
            1.0
            - radius_squared
        )

        return values.astype(
            np.float64,
            copy=False,
        )

    def f_fn(
        x: np.ndarray,
    ) -> np.ndarray:
        points = _ensure_2d(
            x,
            name="x",
        )

        return np.full(
            (points.shape[0],),
            source_amplitude,
            dtype=np.float64,
        )

    def q_fn(
        x: np.ndarray,
        normals: np.ndarray,
    ) -> np.ndarray:
        points, normal_vectors = (
            _prepare_points_and_normals(
                x,
                normals,
            )
        )

        # grad(u) = -(c/2) [x, y].
        gradient = (
            -0.5
            * source_amplitude
            * points
        )

        return np.sum(
            gradient
            * normal_vectors,
            axis=1,
        )

    return ProblemSample(
        case_name="case1",
        geometry_type="disk",
        params=params,
        u_fn=u_fn,
        f_fn=f_fn,
        q_fn=q_fn,
    )


# =========================================================
# Case 2
# =========================================================

def sample_case2(
    rng: np.random.Generator,
) -> ProblemSample:
    """
    Polynomial Poisson problem on the unit disk.

    Exact solution:

        u(x, y)
        =
        (1 - x^2 - y^2)
        (a + bx + cy).

    Governing equation:

        -Delta u
        =
        4a + 8bx + 8cy.

    The Dirichlet value is zero on the unit-circle boundary.
    """
    rng = _validate_rng(
        rng
    )

    parameter_a = float(
        rng.uniform(
            0.2,
            1.5,
        )
    )

    parameter_b = float(
        rng.uniform(
            -1.0,
            1.0,
        )
    )

    parameter_c = float(
        rng.uniform(
            -1.0,
            1.0,
        )
    )

    params = _build_params(
        parameter_a,
        parameter_b,
        parameter_c,
    )

    def u_fn(
        x: np.ndarray,
    ) -> np.ndarray:
        points = _ensure_2d(
            x,
            name="x",
        )

        xx = points[:, 0]
        yy = points[:, 1]

        polynomial = (
            parameter_a
            + parameter_b * xx
            + parameter_c * yy
        )

        radial_factor = (
            1.0
            - xx**2
            - yy**2
        )

        return (
            radial_factor
            * polynomial
        )

    def f_fn(
        x: np.ndarray,
    ) -> np.ndarray:
        points = _ensure_2d(
            x,
            name="x",
        )

        xx = points[:, 0]
        yy = points[:, 1]

        return (
            4.0 * parameter_a
            + 8.0 * parameter_b * xx
            + 8.0 * parameter_c * yy
        )

    def q_fn(
        x: np.ndarray,
        normals: np.ndarray,
    ) -> np.ndarray:
        points, normal_vectors = (
            _prepare_points_and_normals(
                x,
                normals,
            )
        )

        xx = points[:, 0]
        yy = points[:, 1]

        polynomial = (
            parameter_a
            + parameter_b * xx
            + parameter_c * yy
        )

        radial_factor = (
            1.0
            - xx**2
            - yy**2
        )

        # Exact gradient throughout the domain:
        #
        # du/dx = -2x p + (1-r^2)b
        # du/dy = -2y p + (1-r^2)c
        gradient_x = (
            -2.0 * xx * polynomial
            + radial_factor * parameter_b
        )

        gradient_y = (
            -2.0 * yy * polynomial
            + radial_factor * parameter_c
        )

        gradient = np.stack(
            [
                gradient_x,
                gradient_y,
            ],
            axis=1,
        )

        return np.sum(
            gradient
            * normal_vectors,
            axis=1,
        )

    return ProblemSample(
        case_name="case2",
        geometry_type="disk",
        params=params,
        u_fn=u_fn,
        f_fn=f_fn,
        q_fn=q_fn,
    )


# =========================================================
# Case 3
# =========================================================

def sample_case3(
    rng: np.random.Generator,
) -> ProblemSample:
    """
    Trigonometric Poisson problem on the unit square.

    Exact solution:

        u(x, y)
        =
        A sin(m*pi*x) sin(n*pi*y).

    Governing equation:

        -Delta u
        =
        A*pi^2*(m^2+n^2)
        sin(m*pi*x) sin(n*pi*y).

    The Dirichlet value is zero on the complete square boundary.
    """
    rng = _validate_rng(
        rng
    )

    amplitude = float(
        rng.uniform(
            0.4,
            1.8,
        )
    )

    frequency_m = int(
        rng.integers(
            1,
            5,
        )
    )

    frequency_n = int(
        rng.integers(
            1,
            5,
        )
    )

    params = _build_params(
        amplitude,
        float(frequency_m),
        float(frequency_n),
    )

    def u_fn(
        x: np.ndarray,
    ) -> np.ndarray:
        points = _ensure_2d(
            x,
            name="x",
        )

        xx = points[:, 0]
        yy = points[:, 1]

        return (
            amplitude
            * np.sin(
                frequency_m
                * math.pi
                * xx
            )
            * np.sin(
                frequency_n
                * math.pi
                * yy
            )
        )

    def f_fn(
        x: np.ndarray,
    ) -> np.ndarray:
        points = _ensure_2d(
            x,
            name="x",
        )

        xx = points[:, 0]
        yy = points[:, 1]

        factor = (
            amplitude
            * math.pi**2
            * (
                frequency_m**2
                + frequency_n**2
            )
        )

        return (
            factor
            * np.sin(
                frequency_m
                * math.pi
                * xx
            )
            * np.sin(
                frequency_n
                * math.pi
                * yy
            )
        )

    def q_fn(
        x: np.ndarray,
        normals: np.ndarray,
    ) -> np.ndarray:
        points, normal_vectors = (
            _prepare_points_and_normals(
                x,
                normals,
            )
        )

        xx = points[:, 0]
        yy = points[:, 1]

        gradient_x = (
            amplitude
            * frequency_m
            * math.pi
            * np.cos(
                frequency_m
                * math.pi
                * xx
            )
            * np.sin(
                frequency_n
                * math.pi
                * yy
            )
        )

        gradient_y = (
            amplitude
            * frequency_n
            * math.pi
            * np.sin(
                frequency_m
                * math.pi
                * xx
            )
            * np.cos(
                frequency_n
                * math.pi
                * yy
            )
        )

        gradient = np.stack(
            [
                gradient_x,
                gradient_y,
            ],
            axis=1,
        )

        return np.sum(
            gradient
            * normal_vectors,
            axis=1,
        )

    return ProblemSample(
        case_name="case3",
        geometry_type="square",
        params=params,
        u_fn=u_fn,
        f_fn=f_fn,
        q_fn=q_fn,
    )


# =========================================================
# Problem registry
# =========================================================

CASE_REGISTRY: Dict[
    str,
    Callable[
        [np.random.Generator],
        ProblemSample,
    ],
] = {
    "case1": sample_case1,
    "case2": sample_case2,
    "case3": sample_case3,
}
