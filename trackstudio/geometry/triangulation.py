"""Metric multi-camera 3D triangulation utilities for Peleng.

Extrinsics map world coordinates to camera coordinates:
    X_camera = R @ X_world + t
Projection matrix: P = K @ [R | t]. Translation uses the chosen world unit,
preferably metres. Observations are pixels in the original distorted image.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CameraModel:
    """Calibrated pinhole camera with world-to-camera extrinsics."""

    camera_id: int
    intrinsic: FloatArray
    rotation: FloatArray
    translation: FloatArray
    distortion: FloatArray | None = None

    def __post_init__(self) -> None:
        intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64).reshape(-1)
        distortion = None if self.distortion is None else np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if intrinsic.shape != (3, 3):
            raise ValueError("intrinsic must have shape (3, 3)")
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        if translation.shape != (3,):
            raise ValueError("translation must have shape (3,)")
        if not all(np.isfinite(value).all() for value in (intrinsic, rotation, translation)):
            raise ValueError("camera parameters must be finite")
        if abs(float(np.linalg.det(intrinsic))) < 1e-12:
            raise ValueError("intrinsic matrix must be invertible")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError("rotation determinant must be +1")
        if distortion is not None and not np.isfinite(distortion).all():
            raise ValueError("distortion coefficients must be finite")
        object.__setattr__(self, "intrinsic", intrinsic)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "distortion", distortion)

    @property
    def projection_matrix(self) -> FloatArray:
        """Return P = K [R | t]."""
        return self.intrinsic @ np.column_stack((self.rotation, self.translation))

    @property
    def center_world(self) -> FloatArray:
        """Return camera centre C = -R.T @ t in world coordinates."""
        return -self.rotation.T @ self.translation

    def undistort_pixel(self, pixel: tuple[float, float]) -> FloatArray:
        """Undistort a pixel while retaining pixel units and the K matrix."""
        point = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
        return cv2.undistortPoints(point, self.intrinsic, self.distortion, P=self.intrinsic).reshape(2)


@dataclass(frozen=True)
class PointObservation:
    """One 2D observation of the same object in a camera image."""

    camera_id: int
    pixel: tuple[float, float]
    confidence: float = 1.0


@dataclass(frozen=True)
class TriangulationResult:
    """Estimated world point and reprojection diagnostics."""

    point_xyz: tuple[float, float, float]
    inlier_camera_ids: tuple[int, ...]
    reprojection_errors_px: dict[int, float]
    mean_reprojection_error_px: float
    max_reprojection_error_px: float


def project_world_point(point_xyz: FloatArray, camera: CameraModel) -> FloatArray:
    """Project a world point into a camera, including lens distortion."""
    point = np.asarray(point_xyz, dtype=np.float64).reshape(1, 1, 3)
    rvec, _ = cv2.Rodrigues(camera.rotation)
    projected, _ = cv2.projectPoints(point, rvec, camera.translation, camera.intrinsic, camera.distortion)
    return projected.reshape(2)


def _triangulate_pair(first: CameraModel, first_pixel: FloatArray,
                      second: CameraModel, second_pixel: FloatArray) -> FloatArray | None:
    homogeneous = cv2.triangulatePoints(
        first.projection_matrix, second.projection_matrix,
        first_pixel.reshape(2, 1), second_pixel.reshape(2, 1),
    ).reshape(4)
    if not np.isfinite(homogeneous).all() or abs(float(homogeneous[3])) < 1e-12:
        return None
    point = homogeneous[:3] / homogeneous[3]
    if not np.isfinite(point).all():
        return None
    for camera in (first, second):
        if (camera.rotation @ point + camera.translation)[2] <= 0:
            return None
    return point


def triangulate_point(
    observations: list[PointObservation],
    cameras: dict[int, CameraModel],
    *,
    reprojection_threshold_px: float = 3.0,
    min_confidence: float = 0.0,
) -> TriangulationResult | None:
    """Triangulate a point from synchronized views with pairwise outlier rejection.

    Uses pair hypotheses and reprojection consensus, then recomputes the point
    from all inliers. Returns None if fewer than two views agree. All views
    must describe the same physical object at the same timestamp.
    """
    if reprojection_threshold_px <= 0:
        raise ValueError("reprojection_threshold_px must be positive")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    usable: list[tuple[PointObservation, CameraModel, FloatArray]] = []
    seen_camera_ids: set[int] = set()
    for observation in observations:
        if observation.camera_id in seen_camera_ids:
            raise ValueError(f"duplicate observation for camera {observation.camera_id}")
        seen_camera_ids.add(observation.camera_id)
        if observation.camera_id not in cameras:
            raise ValueError(f"camera {observation.camera_id} has no calibration")
        if not np.isfinite(observation.confidence) or not 0 <= observation.confidence <= 1:
            raise ValueError("observation confidence must be in [0, 1]")
        pixel = np.asarray(observation.pixel, dtype=np.float64)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("pixel must contain two finite coordinates")
        if observation.confidence < min_confidence:
            continue
        camera = cameras[observation.camera_id]
        usable.append((observation, camera, camera.undistort_pixel((float(pixel[0]), float(pixel[1])))))

    if len(usable) < 2:
        return None

    best_inliers: list[int] = []
    best_point: FloatArray | None = None
    best_error = float("inf")
    for first_idx, second_idx in combinations(range(len(usable)), 2):
        _, first_cam, first_pixel = usable[first_idx]
        _, second_cam, second_pixel = usable[second_idx]
        candidate = _triangulate_pair(first_cam, first_pixel, second_cam, second_pixel)
        if candidate is None:
            continue
        errors = [
            float(np.linalg.norm(project_world_point(candidate, camera) - np.asarray(observation.pixel)))
            for observation, camera, _ in usable
        ]
        inliers = [idx for idx, error in enumerate(errors) if error <= reprojection_threshold_px]
        if len(inliers) < 2:
            continue
        median_error = float(np.median([errors[idx] for idx in inliers]))
        if len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers) and median_error < best_error):
            best_inliers, best_point, best_error = inliers, candidate, median_error

    if best_point is None or len(best_inliers) < 2:
        return None

    if len(best_inliers) > 2:
        rows: list[FloatArray] = []
        for idx in best_inliers:
            _, camera, pixel = usable[idx]
            projection = camera.projection_matrix
            rows.append(pixel[0] * projection[2] - projection[0])
            rows.append(pixel[1] * projection[2] - projection[1])
        _, _, vh = np.linalg.svd(np.concatenate(rows, axis=0))
        homogeneous = vh[-1]
        if abs(float(homogeneous[3])) < 1e-12:
            return None
        point = homogeneous[:3] / homogeneous[3]
    else:
        point = best_point

    for idx in best_inliers:
        camera = usable[idx][1]
        if (camera.rotation @ point + camera.translation)[2] <= 0:
            return None

    errors_by_camera = {
        usable[idx][0].camera_id: float(
            np.linalg.norm(project_world_point(point, usable[idx][1]) - np.asarray(usable[idx][0].pixel))
        )
        for idx in best_inliers
    }
    errors = list(errors_by_camera.values())
    return TriangulationResult(
        point_xyz=(float(point[0]), float(point[1]), float(point[2])),
        inlier_camera_ids=tuple(usable[idx][0].camera_id for idx in best_inliers),
        reprojection_errors_px=errors_by_camera,
        mean_reprojection_error_px=float(np.mean(errors)),
        max_reprojection_error_px=float(np.max(errors)),
    )


__all__ = [
    "CameraModel", "PointObservation", "TriangulationResult",
    "project_world_point", "triangulate_point",
]
