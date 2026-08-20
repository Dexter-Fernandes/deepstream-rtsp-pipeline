import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.ground_plane import (
    Homography,
    foot_pixel_sigma,
    foot_point,
    is_foot_reliable,
    load_homography,
    project_point,
    projection_jacobian,
    projection_sigma,
)


def test_foot_point_uses_the_bottom_centre_of_the_box():
    assert foot_point(10, 20, 30, 40) == pytest.approx((25.0, 60.0))


def test_project_point_applies_perspective_division():
    matrix = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -2.0], [0.1, 0.0, 1.0]])

    assert project_point(matrix, 5.0, 4.0) == pytest.approx((11.0 / 1.5, 10.0 / 1.5))


def test_projection_jacobian_is_the_analytic_perspective_derivative():
    matrix = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -2.0], [0.1, 0.0, 1.0]])

    jacobian = projection_jacobian(matrix, 5.0, 4.0)

    np.testing.assert_allclose(jacobian, [[1.9 / 2.25, 0.0], [-1.0 / 2.25, 2.0]])


def test_projection_sigma_propagates_pixel_noise_with_the_worst_axis_scale():
    matrix = np.array([[0.02, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]])

    assert projection_sigma(matrix, 100.0, 200.0, pixel_sigma=5.0) == pytest.approx(0.1)


def test_foot_pixel_sigma_scales_with_visible_box_height_and_has_a_floor():
    assert foot_pixel_sigma(10, 20, 30, 100, image_height=1080) == pytest.approx(10.0)
    assert foot_pixel_sigma(10, 20, 3, 10, image_height=1080) == pytest.approx(2.0)


def test_foot_pixel_sigma_uses_width_derived_height_when_the_top_is_truncated():
    sigma = foot_pixel_sigma(100, 0, 40, 20, image_height=1080)

    assert sigma == pytest.approx(10.0)


def test_top_truncation_does_not_make_the_foot_unreliable():
    assert is_foot_reliable(20, 0, 30, 100, image_width=200, image_height=200)


def test_bottom_truncation_makes_the_foot_unreliable():
    assert not is_foot_reliable(20, 100, 30, 99, image_width=200, image_height=200)


@pytest.mark.parametrize(("left", "width"), [(1, 30), (170, 29)])
def test_horizontal_truncation_makes_the_foot_unreliable(left: float, width: float):
    assert not is_foot_reliable(left, 20, width, 100, image_width=200, image_height=200)


def test_load_homography_restores_calibration_and_fit_provenance(tmp_path: Path):
    path = tmp_path / "homography_C1.json"
    path.write_text(
        json.dumps(
            {
                "source_id": 0,
                "camera": "C1",
                "matrix": [[1, 0, 2], [0, 1, 3], [0, 0, 1]],
                "image_width": 1920,
                "image_height": 1080,
                "model": "single_plane_homography",
                "fit": {"seed": 0, "holdout_median_m": 0.02},
            }
        )
    )

    calibration = load_homography(path)

    assert isinstance(calibration, Homography)
    assert calibration.source_id == 0
    assert calibration.camera == "C1"
    np.testing.assert_allclose(calibration.matrix, [[1, 0, 2], [0, 1, 3], [0, 0, 1]])
    assert calibration.fit == {"seed": 0, "holdout_median_m": 0.02}
