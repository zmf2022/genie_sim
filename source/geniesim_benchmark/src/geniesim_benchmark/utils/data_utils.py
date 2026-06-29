# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()  # Create singleton instance


def interpolate_pose_trajectory_rel(pose1, pose2, steps):
    """Return a list of incremental poses between two poses.

    Args:
        pose1 (np.ndarray): current pose, 4x4 homogeneous transform matrix
        pose2 (np.ndarray): desired pose, 4x4 homogeneous transform matrix

    Returns:
        increments (list): (N, 6) list, each element = position increment (mm) + euler-xyz increment (rad).
    """

    # Calculate the absolute path
    rot1 = R.from_matrix(pose1[:3, :3])
    rot2 = R.from_matrix(pose2[:3, :3])
    quat1 = rot1.as_quat()
    quat2 = rot2.as_quat()
    translation_path = np.linspace(pose1[:3], pose2[:3], steps)
    key_times = [0, 1]
    key_rots = R.from_quat(np.stack([quat1, quat2]))
    slerp = Slerp(key_times, key_rots)
    times = np.linspace(0, 1, steps)
    interp_rots = slerp(times)
    eulers = interp_rots.as_euler("xyz", degrees=False)

    # Calculate increments
    translation_increments = np.diff(translation_path, axis=0) * 1000.0
    euler_increments = np.diff(eulers, axis=0)

    # Combine increments
    increments = np.concatenate([translation_increments, euler_increments], axis=1)

    # Add initial zero increment for the first step
    initial_increment = np.zeros((1, 6))
    increments = np.vstack([initial_increment, increments])

    return increments.tolist()


def interpolate_pose_trajectory(pose1, pose2, num_interpolations):
    # interpolate rotation
    rot1 = R.from_matrix(pose1[:3, :3])
    rot2 = R.from_matrix(pose2[:3, :3])
    quat1 = rot1.as_quat()
    quat2 = rot2.as_quat()
    times = [0, 1]
    slerp = Slerp(times, R.from_quat([quat1, quat2]))
    interp_times = np.linspace(0, 1, num_interpolations)
    interp_rots = slerp(interp_times)
    interp_matrices = interp_rots.as_matrix()  # (num_interpolations, 3, 3)
    # interpolate translation
    interp_xyz = np.linspace(pose1[:3, 3], pose2[:3, 3], num_interpolations)  # (num_interpolations, 3)

    interp_poses = np.tile(np.eye(4), (num_interpolations, 1, 1))
    interp_poses[:, :3, :3] = interp_matrices
    interp_poses[:, :3, 3] = interp_xyz
    return interp_poses


class CameraInfo:
    """Camera intrisics for point cloud creation."""

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def move_camera_to_object_center(object_pose_camera, camera_pose_world):
    # get tran & rot
    R_cw = camera_pose_world[:3, :3]
    t_cw = camera_pose_world[:3, 3]

    object_pos_camera = object_pose_camera[:3, 3]

    object_pos_world = R_cw @ object_pos_camera + t_cw

    z_distance = object_pos_camera[2]

    new_camera_pos_world = object_pos_world - R_cw @ np.array([0, 0, z_distance])

    new_camera_pose_world = np.eye(4)
    new_camera_pose_world[:3, :3] = R_cw
    new_camera_pose_world[:3, 3] = -new_camera_pos_world

    return new_camera_pose_world


def create_point_cloud_from_depth_image(depth, cam_info, organized=True):
    """Generate point cloud using depth image only.

    Input:
        depth: [numpy.ndarray, (H,W), numpy.float32]
            depth image
        camera: [CameraInfo]
            camera intrinsics
        organized: bool
            whether to keep the cloud in image shape (H,W,3)

    Output:
        cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
            generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    if not isinstance(cam_info, CameraInfo):
        camera = CameraInfo(
            width=cam_info["W"],
            height=cam_info["H"],
            fx=cam_info["K"][0][0],
            fy=cam_info["K"][1][1],
            cx=cam_info["K"][0][2],
            cy=cam_info["K"][1][2],
            scale=cam_info["scale"],
        )
    else:
        camera = cam_info
    assert depth.shape[0] == camera.height and depth.shape[1] == camera.width
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def transform_point_cloud_numpy(cloud, transform, format="4x4"):
    """Transform points to new coordinates with transformation matrix.

    Input:
        cloud: [np.ndarray, (N,3), np.float32]
            points in original coordinates
        transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
            transformation matrix, could be rotation only or rotation+translation
        format: [string, '3x3'/'3x4'/'4x4']
            the shape of transformation matrix
            '3x3' --> rotation matrix
            '3x4'/'4x4' --> rotation matrix + translation matrix

    Output:
        cloud_transformed: [np.ndarray, (N,3), np.float32]
            points in new coordinates
    """
    if not (format == "3x3" or format == "4x4" or format == "3x4"):
        raise ValueError("Unknown transformation format, only support '3x3' or '4x4' or '3x4'.")
    if format == "3x3":
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == "4x4" or format == "3x4":
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def compute_point_dists(A, B):
    """Compute pair-wise point distances in two matrices.

    Input:
        A: [np.ndarray, (N,3), np.float32]
            point cloud A
        B: [np.ndarray, (M,3), np.float32]
            point cloud B

    Output:
        dists: [np.ndarray, (N,M), np.float32]
            distance matrix
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A - B, axis=-1)
    return dists


def mask2bbox(mask):
    nonzero_indices = np.nonzero(mask)
    min_y, min_x = np.min(nonzero_indices, axis=1)
    max_y, max_x = np.max(nonzero_indices, axis=1)
    bbox = [min_x, min_y, max_x, max_y]
    return bbox


def get_3d_center_of_rgbd_mask(masks, depth, cam_info, c2w):
    H, W = depth.shape[:2]
    pts = create_point_cloud_from_depth_image(depth, cam_info)
    pts_world = transform_point_cloud_numpy(pts.reshape(-1, 3), c2w).reshape(H, W, 3)
    depth_mask = (depth > 0.1) & (depth < 2)
    centers = []
    for id in masks:
        mask = masks[id] & depth_mask
        if mask.sum() == 0:
            center_xyz = np.array([np.inf, np.inf, np.inf])
        else:
            xyz = pts_world[mask]
            center_xyz = (np.percentile(xyz, 98, axis=0) + np.percentile(xyz, 2, axis=0)) / 2
        centers.append(center_xyz)
    centers = np.stack(centers)
    return centers


def match_obj_with_new_view(target_xyz, masks, depth, cam_info, c2w, **kwargs):
    obj_centers = get_3d_center_of_rgbd_mask(masks, depth, cam_info, c2w)
    diff = np.linalg.norm(obj_centers - target_xyz, axis=1)
    min_index = np.argmin(diff)
    return min_index


def pose_difference_batch(pose1, pose2):
    # Ensure pose2 is a 4x4 matrix
    assert pose2.shape == (4, 4), "pose2 must be a 4x4 matrix"

    # Extract positions
    positions1 = pose1[:, :3, 3]
    position2 = pose2[:3, 3]

    # Calculate Euclidean distances for positions
    position_distances = np.linalg.norm(positions1 - position2, axis=1)

    # Extract rotation matrices
    rotations1 = pose1[:, :3, :3]
    rotation2 = pose2[:3, :3]

    # Convert to rotation objects
    r1 = R.from_matrix(rotations1)
    r2 = R.from_matrix(rotation2)

    # Calculate rotation differences
    relative_rotations = r1.inv() * r2
    angle_differences = relative_rotations.magnitude()

    return position_distances, np.degrees(angle_differences)


def pose_difference(pose1, pose2):
    # position
    position1 = pose1[:3, 3]
    position2 = pose2[:3, 3]

    # distance
    position_distance = np.linalg.norm(position1 - position2)

    # roation matrix
    rotation1 = pose1[:3, :3]
    rotation2 = pose2[:3, :3]

    # roation matrix angle diff
    r1 = R.from_matrix(rotation1)
    r2 = R.from_matrix(rotation2)

    # angle diff
    relative_rotation = r1.inv() * r2
    angle_difference = relative_rotation.magnitude()

    return position_distance, np.degrees(angle_difference)


def vector_difference(pose1, pose2, vector=np.array([0, 0, 1])):
    rotation1 = pose1[:3, :3]
    rotation2 = pose2[:3, :3]

    transformed_vector1 = rotation1 @ vector
    transformed_vector2 = rotation2 @ vector

    dot_product = np.dot(transformed_vector1, transformed_vector2)
    norms_product = np.linalg.norm(transformed_vector1) * np.linalg.norm(transformed_vector2)
    cos_angle = dot_product / norms_product

    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_difference = np.arccos(cos_angle)

    return np.degrees(angle_difference)


def vector_difference_batch(pose1, pose2, vector=np.array([0, 0, 1])):
    assert pose2.shape == (4, 4), "pose2 must be a 4x4 matrix"

    rotations1 = pose1[:, :3, :3]
    rotation2 = pose2[:3, :3]

    transformed_vectors1 = np.einsum("bij,j->bi", rotations1, vector)
    transformed_vector2 = rotation2 @ vector

    dot_products = np.einsum("bi,i->b", transformed_vectors1, transformed_vector2)
    norms_product1 = np.linalg.norm(transformed_vectors1, axis=1)
    norm_product2 = np.linalg.norm(transformed_vector2)
    norms_products = norms_product1 * norm_product2

    cos_angles = dot_products / norms_products

    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    angle_differences = np.arccos(cos_angles)

    return np.degrees(angle_differences)
