from __future__ import annotations

"""PyTorch Geometric dataset loader for generated boundary-flux samples."""


from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset


_VALID_SPLITS: Final[set[str]] = {
    "train",
    "val",
    "test",
}


# =========================================================
# NumPy conversion and validation utilities
# =========================================================

def _require_key(
    archive: np.lib.npyio.NpzFile,
    key: str,
    file_path: Path,
) -> None:
    if key not in archive.files:
        raise KeyError(
            f"Required key {key!r} is missing from "
            f"{file_path}"
        )


def _read_string(
    archive: np.lib.npyio.NpzFile,
    key: str,
    file_path: Path,
) -> str:
    """
    Read a scalar string stored in an NPZ archive.

    Supports NumPy Unicode strings and byte strings.
    """
    _require_key(
        archive,
        key,
        file_path,
    )

    array = np.asarray(
        archive[key]
    )

    if array.size != 1:
        raise ValueError(
            f"{key!r} in {file_path} must contain exactly "
            f"one string, but has shape {array.shape}"
        )

    value = array.reshape(-1)[0]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bytes):
        value = value.decode(
            "utf-8"
        )

    return str(value)


def _read_float_array(
    archive: np.lib.npyio.NpzFile,
    key: str,
    file_path: Path,
) -> np.ndarray:
    _require_key(
        archive,
        key,
        file_path,
    )

    return np.ascontiguousarray(
        np.asarray(
            archive[key],
            dtype=np.float32,
        )
    )


def _read_int_array(
    archive: np.lib.npyio.NpzFile,
    key: str,
    file_path: Path,
) -> np.ndarray:
    _require_key(
        archive,
        key,
        file_path,
    )

    return np.ascontiguousarray(
        np.asarray(
            archive[key],
            dtype=np.int64,
        )
    )


def _read_bool_array(
    archive: np.lib.npyio.NpzFile,
    key: str,
    file_path: Path,
) -> np.ndarray:
    _require_key(
        archive,
        key,
        file_path,
    )

    return np.ascontiguousarray(
        np.asarray(
            archive[key],
            dtype=np.bool_,
        )
    )


def _validate_finite(
    name: str,
    array: np.ndarray,
    file_path: Path,
) -> None:
    if not np.all(
        np.isfinite(array)
    ):
        raise FloatingPointError(
            f"{name} in {file_path} contains "
            "NaN or infinite values"
        )


def _to_float_tensor(
    array: np.ndarray,
) -> Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(
            array,
            dtype=np.float32,
        )
    )


def _to_long_tensor(
    array: np.ndarray,
) -> Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(
            array,
            dtype=np.int64,
        )
    )


def _to_bool_tensor(
    array: np.ndarray,
) -> Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(
            array,
            dtype=np.bool_,
        )
    )


# =========================================================
# Dataset
# =========================================================

class BoundaryFluxDataset(Dataset):
    """
    Dataset for the single-sample boundary-flux fitting experiment.

    Basic sample fields
    -------------------
    x:
        Boundary-node feature matrix with shape [N, F].

    edge_index:
        Directed graph edges with shape [2, E].

    y:
        Exact Neumann target with shape [N].

    When full_load=True, the geometrical, boundary, and domain fields
    required by the BEM testing routine are also attached to the Data
    object.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        full_load: bool = False,
    ) -> None:
        root_path = Path(
            root
        ).expanduser()

        split = str(
            split
        ).strip().lower()

        if split not in _VALID_SPLITS:
            valid_splits = ", ".join(
                sorted(_VALID_SPLITS)
            )

            raise ValueError(
                f"Unsupported split={split!r}. "
                f"Expected one of: {valid_splits}"
            )

        super().__init__(
            root=str(root_path)
        )

        self.root_dir = root_path
        self.split = split
        self.full_load = bool(
            full_load
        )

        self.split_dir = (
            self.root_dir
            / self.split
        )

        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset split directory does not exist: "
                f"{self.split_dir}"
            )

        # Only load generated sample files. This prevents unrelated NPZ
        # files in the directory from being interpreted as dataset items.
        self.files: list[Path] = sorted(
            self.split_dir.glob(
                "sample_*.npz"
            )
        )

        if not self.files:
            raise FileNotFoundError(
                f"No sample_*.npz files found in "
                f"{self.split_dir}"
            )

    def len(self) -> int:
        return len(
            self.files
        )

    def _normalize_index(
        self,
        idx: int,
    ) -> int:
        if isinstance(
            idx,
            np.integer,
        ):
            idx = int(
                idx
            )

        if not isinstance(
            idx,
            int,
        ):
            raise TypeError(
                f"Dataset index must be an integer, "
                f"but received {type(idx).__name__}"
            )

        if idx < 0:
            idx += len(
                self.files
            )

        if idx < 0 or idx >= len(
            self.files
        ):
            raise IndexError(
                f"Dataset index {idx} is outside the valid "
                f"range [0, {len(self.files) - 1}]"
            )

        return idx

    def get(
        self,
        idx: int,
    ) -> Data:
        idx = self._normalize_index(
            idx
        )

        file_path = self.files[
            idx
        ]

        # The generated files contain only standard NumPy arrays and
        # scalar strings, so pickle loading is neither needed nor desired.
        with np.load(
            file_path,
            allow_pickle=False,
        ) as archive:
            data = self._build_basic_data(
                archive=archive,
                file_path=file_path,
            )

            if self.full_load:
                self._attach_full_data(
                    data=data,
                    archive=archive,
                    file_path=file_path,
                )

        # Useful metadata for diagnostics. These attributes do not alter
        # the model input or batching behavior.
        data.sample_index = int(
            idx
        )
        data.sample_path = str(
            file_path
        )

        return data

    def _build_basic_data(
        self,
        archive: np.lib.npyio.NpzFile,
        file_path: Path,
    ) -> Data:
        node_features = _read_float_array(
            archive,
            "node_features",
            file_path,
        )

        edge_index = _read_int_array(
            archive,
            "edge_index",
            file_path,
        )

        q_exact = _read_float_array(
            archive,
            "q_exact",
            file_path,
        ).reshape(-1)

        if node_features.ndim != 2:
            raise ValueError(
                f"node_features in {file_path} must have shape "
                f"[N, F], but received {node_features.shape}"
            )

        num_nodes = int(
            node_features.shape[0]
        )

        num_features = int(
            node_features.shape[1]
        )

        if num_nodes <= 0:
            raise ValueError(
                f"node_features in {file_path} contains no nodes"
            )

        if num_features <= 0:
            raise ValueError(
                f"node_features in {file_path} contains no features"
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index in {file_path} must have shape "
                f"[2, E], but received {edge_index.shape}"
            )

        if q_exact.shape[0] != num_nodes:
            raise ValueError(
                f"q_exact in {file_path} contains "
                f"{q_exact.shape[0]} values, but node_features "
                f"contains {num_nodes} nodes"
            )

        _validate_finite(
            "node_features",
            node_features,
            file_path,
        )

        _validate_finite(
            "q_exact",
            q_exact,
            file_path,
        )

        if edge_index.size > 0:
            minimum_index = int(
                edge_index.min()
            )

            maximum_index = int(
                edge_index.max()
            )

            if minimum_index < 0:
                raise ValueError(
                    f"edge_index in {file_path} contains a "
                    f"negative node index: {minimum_index}"
                )

            if maximum_index >= num_nodes:
                raise ValueError(
                    f"edge_index in {file_path} contains node "
                    f"index {maximum_index}, but the graph only "
                    f"contains {num_nodes} nodes"
                )

        case_name = _read_string(
            archive,
            "case_name",
            file_path,
        )

        geometry_type = _read_string(
            archive,
            "geometry_type",
            file_path,
        )

        data = Data(
            x=_to_float_tensor(
                node_features
            ),
            edge_index=_to_long_tensor(
                edge_index
            ),
            y=_to_float_tensor(
                q_exact
            ),
            num_nodes=num_nodes,
        )

        data.case_name = case_name
        data.geometry_type = geometry_type

        return data

    def _attach_full_data(
        self,
        data: Data,
        archive: np.lib.npyio.NpzFile,
        file_path: Path,
    ) -> None:
        """
        Attach boundary geometry and optional domain fields.

        Boundary fields are required when full_load=True. Domain fields
        are loaded as a complete group when present.
        """
        num_nodes = int(
            data.num_nodes
        )

        required_boundary_keys = (
            "coords",
            "normals",
            "ds",
            "u_bc",
            "f_bnd",
            "params",
        )

        for key in required_boundary_keys:
            _require_key(
                archive,
                key,
                file_path,
            )

        coords = _read_float_array(
            archive,
            "coords",
            file_path,
        )

        normals = _read_float_array(
            archive,
            "normals",
            file_path,
        )

        ds = _read_float_array(
            archive,
            "ds",
            file_path,
        ).reshape(-1)

        u_bc = _read_float_array(
            archive,
            "u_bc",
            file_path,
        ).reshape(-1)

        f_bnd = _read_float_array(
            archive,
            "f_bnd",
            file_path,
        ).reshape(-1)

        params = _read_float_array(
            archive,
            "params",
            file_path,
        ).reshape(-1)

        if coords.shape != (
            num_nodes,
            2,
        ):
            raise ValueError(
                f"coords in {file_path} must have shape "
                f"[{num_nodes}, 2], but received {coords.shape}"
            )

        if normals.shape != coords.shape:
            raise ValueError(
                f"normals in {file_path} must have shape "
                f"{coords.shape}, but received {normals.shape}"
            )

        if ds.shape[0] != num_nodes:
            raise ValueError(
                f"ds in {file_path} contains {ds.shape[0]} "
                f"values, but {num_nodes} are required"
            )

        if u_bc.shape[0] != num_nodes:
            raise ValueError(
                f"u_bc in {file_path} contains {u_bc.shape[0]} "
                f"values, but {num_nodes} are required"
            )

        if f_bnd.shape[0] != num_nodes:
            raise ValueError(
                f"f_bnd in {file_path} contains "
                f"{f_bnd.shape[0]} values, but {num_nodes} "
                "are required"
            )

        if params.size == 0:
            raise ValueError(
                f"params in {file_path} is empty"
            )

        _validate_finite(
            "coords",
            coords,
            file_path,
        )

        _validate_finite(
            "normals",
            normals,
            file_path,
        )

        _validate_finite(
            "ds",
            ds,
            file_path,
        )

        _validate_finite(
            "u_bc",
            u_bc,
            file_path,
        )

        _validate_finite(
            "f_bnd",
            f_bnd,
            file_path,
        )

        _validate_finite(
            "params",
            params,
            file_path,
        )

        if np.any(
            ds <= 0.0
        ):
            invalid_indices = np.where(
                ds <= 0.0
            )[0]

            raise ValueError(
                f"ds in {file_path} contains non-positive "
                f"values at indices {invalid_indices.tolist()}"
            )

        normal_norms = np.linalg.norm(
            normals,
            axis=1,
        )

        if np.any(
            normal_norms <= 1e-12
        ):
            invalid_indices = np.where(
                normal_norms <= 1e-12
            )[0]

            raise ValueError(
                f"normals in {file_path} contains zero-length "
                f"vectors at indices {invalid_indices.tolist()}"
            )

        data.coords = _to_float_tensor(
            coords
        )

        data.normals = _to_float_tensor(
            normals
        )

        data.ds = _to_float_tensor(
            ds
        )

        data.u_bc = _to_float_tensor(
            u_bc
        )

        data.f_bnd = _to_float_tensor(
            f_bnd
        )

        data.params = _to_float_tensor(
            params
        )

        # q_exact is already available as data.y. Attach an explicit
        # alias for clearer use in diagnostics if needed.
        data.q_exact = data.y.clone()

        domain_keys = (
            "domain_points",
            "domain_mask",
            "domain_shape",
            "u_domain_exact",
            "f_domain",
        )

        present_domain_keys = {
            key
            for key in domain_keys
            if key in archive.files
        }

        if not present_domain_keys:
            return

        missing_domain_keys = (
            set(domain_keys)
            - present_domain_keys
        )

        if missing_domain_keys:
            raise KeyError(
                f"Domain data in {file_path} is incomplete. "
                "Missing keys: "
                + ", ".join(
                    sorted(
                        missing_domain_keys
                    )
                )
            )

        domain_points = _read_float_array(
            archive,
            "domain_points",
            file_path,
        )

        domain_mask = _read_bool_array(
            archive,
            "domain_mask",
            file_path,
        ).reshape(-1)

        domain_shape = _read_int_array(
            archive,
            "domain_shape",
            file_path,
        ).reshape(-1)

        u_domain_exact = _read_float_array(
            archive,
            "u_domain_exact",
            file_path,
        ).reshape(-1)

        f_domain = _read_float_array(
            archive,
            "f_domain",
            file_path,
        ).reshape(-1)

        if (
            domain_points.ndim != 2
            or domain_points.shape[1] != 2
        ):
            raise ValueError(
                f"domain_points in {file_path} must have "
                f"shape [M, 2], but received "
                f"{domain_points.shape}"
            )

        num_domain_points = int(
            domain_points.shape[0]
        )

        if domain_mask.shape[0] != num_domain_points:
            raise ValueError(
                f"domain_mask in {file_path} contains "
                f"{domain_mask.shape[0]} values, but "
                f"domain_points contains {num_domain_points} points"
            )

        if domain_shape.shape[0] != 2:
            raise ValueError(
                f"domain_shape in {file_path} must contain "
                f"two values, but received {domain_shape}"
            )

        expected_grid_size = int(
            np.prod(
                domain_shape
            )
        )

        if expected_grid_size != num_domain_points:
            raise ValueError(
                f"domain_shape={tuple(domain_shape.tolist())} "
                f"contains {expected_grid_size} grid locations, "
                f"but domain_points contains "
                f"{num_domain_points} points"
            )

        if u_domain_exact.shape[0] != num_domain_points:
            raise ValueError(
                f"u_domain_exact in {file_path} contains "
                f"{u_domain_exact.shape[0]} values, but "
                f"{num_domain_points} are required"
            )

        if f_domain.shape[0] != num_domain_points:
            raise ValueError(
                f"f_domain in {file_path} contains "
                f"{f_domain.shape[0]} values, but "
                f"{num_domain_points} are required"
            )

        if not np.any(
            domain_mask
        ):
            raise ValueError(
                f"domain_mask in {file_path} contains no "
                "interior points"
            )

        _validate_finite(
            "domain_points",
            domain_points,
            file_path,
        )

        # NaNs are intentionally permitted outside the interior mask.
        if not np.all(
            np.isfinite(
                u_domain_exact[
                    domain_mask
                ]
            )
        ):
            raise FloatingPointError(
                f"u_domain_exact in {file_path} contains "
                "NaN or infinite values inside the domain mask"
            )

        if not np.all(
            np.isfinite(
                f_domain[
                    domain_mask
                ]
            )
        ):
            raise FloatingPointError(
                f"f_domain in {file_path} contains "
                "NaN or infinite values inside the domain mask"
            )

        data.domain_points = _to_float_tensor(
            domain_points
        )

        data.domain_mask = _to_bool_tensor(
            domain_mask
        )

        data.domain_shape = _to_long_tensor(
            domain_shape
        )

        data.u_domain_exact = _to_float_tensor(
            u_domain_exact
        )

        data.f_domain = _to_float_tensor(
            f_domain
        )
