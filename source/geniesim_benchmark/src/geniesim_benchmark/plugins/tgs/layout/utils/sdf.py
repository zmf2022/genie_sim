# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import trimesh
import numpy as np


def compute_sdf_from_obj_surface(mesh, resolution=2):  # 2mm
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError("Invalid mesh provided.")
    bounds_min, bounds_max = mesh.bounds
    shape = np.ceil((bounds_max - bounds_min) / resolution).astype(int)

    grid = np.mgrid[
        bounds_min[0] : bounds_max[0] : complex(0, shape[0]),
        bounds_min[1] : bounds_max[1] : complex(0, shape[1]),
        bounds_min[2] : bounds_max[2] : complex(0, shape[2]),
    ]
    grid = grid.reshape(3, -1).T

    proximity = trimesh.proximity.ProximityQuery(mesh)
    distances = proximity.signed_distance(grid.astype(np.float32))
    sdf_voxels = distances.reshape(shape)

    return grid, sdf_voxels


def compute_sdf_from_obj(obj_mes, bounds_max, bounds_min, resolution=2):  # 2mm
    shape = np.ceil((np.array(bounds_max) - np.array(bounds_min)) / resolution).astype(int)
    steps = (np.array(bounds_max) - np.array(bounds_min)) / shape
    grid = np.mgrid[
        bounds_min[0] : bounds_max[0] : steps[0],
        bounds_min[1] : bounds_max[1] : steps[1],
        bounds_min[2] : bounds_max[2] : steps[2],
    ]
    grid = grid.reshape(3, -1).T

    proximity = trimesh.proximity.ProximityQuery(obj_mes)
    distances = proximity.signed_distance(grid.astype(np.float32))
    sdf_voxels = distances.reshape(shape)

    return grid, sdf_voxels
