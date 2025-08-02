# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.utils.logger import Logger

logger = Logger()  # Create singleton instance

import omni
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdPhysics, PhysxSchema

from pprint import pprint
import re


def get_articulated_object_prims(articulate_object_name):
    # Get the current USD stage
    stage = omni.usd.get_context().get_stage()
    # List to store prims with both RigidBody and Collider
    articulated_prims = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh" and str(prim.GetPrimPath()).startswith(
            articulate_object_name
        ):
            articulated_prims.append(str(prim.GetPrimPath()))
    return articulated_prims


def is_valid_path(path):
    pattern = r"^/World/Objects/[^/]+$"
    return re.fullmatch(pattern, path) is not None


def get_rigidbody_collider_prims(robot_name, extra_prim_paths=None):
    # Get the current USD stage
    stage = omni.usd.get_context().get_stage()

    # List to store prims with both RigidBody and Collider
    xform_prims = []
    candid_prims = []
    for prim in stage.Traverse():
        if extra_prim_paths is not None and str(prim.GetPrimPath()) in extra_prim_paths:
            candid_prims.append(prim)
            continue

        if prim.GetTypeName() == "Xform":
            if not robot_name:
                xform_prims.append(prim)
            else:
                if robot_name not in str(prim.GetPrimPath()):
                    xform_prims.append(prim)

    valid_prims = []
    for prim in xform_prims:
        if not is_valid_path(str(prim.GetPrimPath())):
            continue
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_attr = prim.GetAttribute("physics:mass")
            if mass_attr.IsValid():
                mass = mass_attr.Get()
                # Ensure mass is a number and > 0
                if isinstance(mass, (float, int)) and mass > 0:
                    valid_prims.append(prim)

    for prim in valid_prims:
        has_cld = set()
        for descendant in stage.Traverse():
            if not str(descendant.GetPrimPath()).startswith(str(prim.GetPrimPath())):
                continue

            has_cld.add(descendant.HasAPI(PhysxSchema.PhysxCollisionAPI))

        has_cld.add(prim.HasAPI(PhysxSchema.PhysxCollisionAPI))

        if has_cld.intersection({True}):
            candid_prims.append(prim)

    return candid_prims


def get_camera_prims(robot_name):
    # Get the current USD stage
    stage = omni.usd.get_context().get_stage()

    # List to store camera prims
    camera_prims = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Camera":
            if not robot_name:
                camera_prims.append(prim)
            else:
                if robot_name in str(prim.GetPrimPath()):
                    camera_prims.append(prim)

    return camera_prims
