# Peleng: metric 3D triangulation

This branch adds a calibrated-camera triangulation primitive to TrackStudio. It complements the existing BEV merger; it does not replace it.

## Coordinate convention

    X_camera = R_world_to_camera @ X_world + t_world_to_camera
    P = K @ [R | t]
    camera_center_world = -R.T @ t

All translations must use one consistent world unit (metres recommended). Image points are pixels in the original image. The utility undistorts pixels before triangulation and reports reprojection errors in pixels.

## Minimal example

    import numpy as np
    from trackstudio.geometry.triangulation import CameraModel, PointObservation, triangulate_point

    K = np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]])
    left = CameraModel(0, K, np.eye(3), np.array([0., 0., 0.]))
    right = CameraModel(1, K, np.eye(3), np.array([-1., 0., 0.]))
    observations = [
        PointObservation(0, (600., 340.)),
        PointObservation(1, (450., 340.)),
    ]
    result = triangulate_point(observations, {0: left, 1: right})
    if result is not None:
        print(result.point_xyz, result.mean_reprojection_error_px)

## Important limitations and integration plan

- Calibration must be measured or estimated first; inconsistent extrinsics cannot produce trustworthy metric coordinates.
- Observations must refer to the same object and synchronized time. Timing errors create motion-dependent position errors.
- This API triangulates one correspondence at a time. The next integration layer must associate TrackStudio tracks across cameras, call this function per global object, and expose XYZ plus reprojection diagnostics through the live API/UI.
- Validate against measured ground-truth points. Report 3D Euclidean MAE/RMSE/P95, per-axis errors, reprojection error, coverage, and end-to-end latency.
- Synthetic tests validate geometry and outlier rejection; they do not replace physical-camera testing.
