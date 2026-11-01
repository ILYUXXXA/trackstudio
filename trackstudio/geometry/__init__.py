"""Geometry algorithms for calibrated multi-camera tracking."""

from .triangulation import CameraModel, PointObservation, TriangulationResult, project_world_point, triangulate_point

__all__ = [
    "CameraModel",
    "PointObservation",
    "TriangulationResult",
    "project_world_point",
    "triangulate_point",
]
