"""Synthetic tests for metric multi-camera triangulation."""

import numpy as np
import pytest

from trackstudio.geometry.triangulation import (
    CameraModel,
    PointObservation,
    project_world_point,
    triangulate_point,
)


def make_camera(camera_id: int, center_x: float) -> CameraModel:
    intrinsic = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    return CameraModel(camera_id, intrinsic, np.eye(3), np.array([-center_x, 0.0, 0.0]))


def observations_for(point: np.ndarray, cameras: dict[int, CameraModel]) -> list[PointObservation]:
    return [
        PointObservation(camera_id, tuple(project_world_point(point, camera)))
        for camera_id, camera in cameras.items()
    ]


def test_triangulates_metric_point_from_two_cameras() -> None:
    cameras = {0: make_camera(0, -1.0), 1: make_camera(1, 1.0)}
    expected = np.array([0.25, -0.15, 6.0])
    result = triangulate_point(observations_for(expected, cameras), cameras)
    assert result is not None
    np.testing.assert_allclose(result.point_xyz, expected, atol=1e-5)
    assert result.mean_reprojection_error_px < 1e-5
    assert set(result.inlier_camera_ids) == {0, 1}


def test_rejects_one_bad_camera_with_three_views() -> None:
    cameras = {0: make_camera(0, -1.0), 1: make_camera(1, 0.0), 2: make_camera(2, 1.0)}
    expected = np.array([0.2, 0.1, 5.0])
    observations = observations_for(expected, cameras)
    observations[2] = PointObservation(2, (100.0, 100.0))
    result = triangulate_point(observations, cameras, reprojection_threshold_px=2.0)
    assert result is not None
    np.testing.assert_allclose(result.point_xyz, expected, atol=1e-5)
    assert set(result.inlier_camera_ids) == {0, 1}


def test_returns_none_when_only_one_view_is_usable() -> None:
    cameras = {0: make_camera(0, 0.0)}
    observation = PointObservation(0, (640.0, 360.0))
    assert triangulate_point([observation], cameras) is None


def test_rejects_duplicate_camera_observations() -> None:
    cameras = {0: make_camera(0, 0.0)}
    observations = [PointObservation(0, (640.0, 360.0)), PointObservation(0, (641.0, 360.0))]
    with pytest.raises(ValueError, match="duplicate observation"):
        triangulate_point(observations, cameras)


def test_rejects_invalid_intrinsic_matrix() -> None:
    with pytest.raises(ValueError, match="shape"):
        CameraModel(0, np.eye(2), np.eye(3), np.zeros(3))
