# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""geniesim_ros — ROS 2 workspace bridge for Genie Sim.

This distribution has a hybrid layout:

* ``src/geniesim_ros/`` — pip-installable Python shim (this file).
  Importable as ``import geniesim_ros``.
* ``src/ros_ws/src/`` — bundled colcon source workspace holding ament
  packages (urdf/xacro/meshes/launch, plus C++ where applicable).
  Build with::

      cd <path-to>/source/geniesim_ros/src/ros_ws
      colcon build
      source install/setup.bash

The pip package and the colcon workspace are intentionally decoupled:
``pip install -e source/geniesim_ros`` only installs the Python shim;
``colcon build`` only builds the ament packages. ``COLCON_IGNORE``
markers prevent colcon from descending into the pip side, and
``[tool.setuptools.packages.find].exclude = ["ros_ws*"]`` prevents
setuptools from descending into the colcon side.

Wheel installs ship the colcon install tree as ``_ros_install.tar.gz``
inside the wheel (tar preserves file modes — a wheel ``package_data``
glob would silently strip the ``0o755`` bit ROS 2 needs on script
entrypoints). The tarball is extracted lazily on first import; see
:mod:`geniesim_ros._bootstrap` for the mechanism. Editable installs
skip extraction entirely — devs build colcon themselves.
"""

from ._bootstrap import ensure_ros_install, install_root, setup_bash_path
from ._version import _resolve_version

__version__: str = _resolve_version("geniesim_ros")
__all__ = ["ensure_ros_install", "install_root", "setup_bash_path", "__version__"]

# Trigger lazy extraction on first import. This is the contract: by the
# time ``import geniesim_ros`` returns, the colcon install tree is on
# disk (if a tarball is bundled) and ready to be sourced. Returns None
# silently when no tarball is bundled AND no pre-staged tree exists
# (editable / source-checkout path) — callers that need the tree should
# check ``install_root()`` and handle ``None`` as "user must build first".
ensure_ros_install()
