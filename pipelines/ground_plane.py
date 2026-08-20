"""Ground-plane projection utilities shared by offline metrics and the live pipeline."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Homography:
    """One camera's image-pixel to world-metre calibration."""

    source_id: int
    camera: str
    matrix: np.ndarray
    image_width: int
    image_height: int
    model: str = "single_plane_homography"
    fit: dict | None = None

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=float, copy=True)
        if matrix.shape != (3, 3):
            raise ValueError(f"homography matrix must have shape (3, 3), got {matrix.shape}")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)


def load_homography(path: str | Path) -> Homography:
    """Load a fitted homography JSON artifact."""
    data = json.loads(Path(path).read_text())
    return Homography(
        source_id=int(data["source_id"]),
        camera=str(data["camera"]),
        matrix=np.asarray(data["matrix"], dtype=float),
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        model=str(data.get("model", "single_plane_homography")),
        fit=data.get("fit"),
    )


def foot_point(
    left: float,
    top: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    """Return the bottom-centre pixel of an image-space bounding box."""
    return float(left + width / 2.0), float(top + height)


def project_point(matrix: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """Project one image pixel through a 3x3 image-to-world homography."""
    matrix = np.asarray(matrix, dtype=float)
    homogeneous = matrix @ np.array([u, v, 1.0], dtype=float)
    denominator = float(homogeneous[2])
    if abs(denominator) <= np.finfo(float).eps:
        raise ValueError("homography projects point to infinity")
    return float(homogeneous[0] / denominator), float(homogeneous[1] / denominator)


def projection_jacobian(matrix: np.ndarray, u: float, v: float) -> np.ndarray:
    """Return the analytic 2x2 derivative of world position with respect to pixels."""
    matrix = np.asarray(matrix, dtype=float)
    x_numerator = matrix[0, 0] * u + matrix[0, 1] * v + matrix[0, 2]
    y_numerator = matrix[1, 0] * u + matrix[1, 1] * v + matrix[1, 2]
    denominator = matrix[2, 0] * u + matrix[2, 1] * v + matrix[2, 2]
    if abs(denominator) <= np.finfo(float).eps:
        raise ValueError("homography projects point to infinity")
    denominator_squared = denominator**2
    return np.array(
        [
            [
                (matrix[0, 0] * denominator - x_numerator * matrix[2, 0])
                / denominator_squared,
                (matrix[0, 1] * denominator - x_numerator * matrix[2, 1])
                / denominator_squared,
            ],
            [
                (matrix[1, 0] * denominator - y_numerator * matrix[2, 0])
                / denominator_squared,
                (matrix[1, 1] * denominator - y_numerator * matrix[2, 1])
                / denominator_squared,
            ],
        ]
    )


def projection_sigma(
    matrix: np.ndarray,
    u: float,
    v: float,
    pixel_sigma: float,
) -> float:
    """Propagate isotropic foot-pixel uncertainty to a conservative metric sigma."""
    jacobian = projection_jacobian(matrix, u, v)
    return float(pixel_sigma * np.linalg.norm(jacobian, ord=2))


def foot_pixel_sigma(
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    image_height: int,
    frac: float = 0.10,
    floor: float = 2.0,
    hw_ratio: float = 2.5,
) -> float:
    """Estimate bottom-centre uncertainty in pixels from pedestrian box scale."""
    effective_height = width * hw_ratio if top <= 2.0 else height
    return float(max(floor, frac * effective_height))


def is_foot_reliable(
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    image_width: int,
    image_height: int,
    margin: float = 2.0,
) -> bool:
    """Return whether a box has an observable bottom-centre foot point."""
    bottom_is_visible = top + height < image_height - margin
    sides_are_visible = left > margin and left + width < image_width - margin
    return bottom_is_visible and sides_are_visible
