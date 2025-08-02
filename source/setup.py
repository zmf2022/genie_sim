# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0


"""Installation script for the 'geniesim' python package."""

# 1. install isaacsim
# reference: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-lab

# 2. pip install -e ./source

# OTHERS:
# conda install -c conda-forge libstdcxx-ng=13

import os
from setuptools import setup

SETUP_PATH = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_PATH = os.path.join(SETUP_PATH, "../requirements.txt")
WHEEL_PATH = os.path.abspath(
    os.path.join(SETUP_PATH, "../3rdparty/ik_solver-0.4.3-cp310-cp310-linux_x86_64.whl")
)


def read_requirements(file_path):
    requirements = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requirements.append(line)
    return requirements


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = read_requirements(REQUIREMENTS_PATH)
INSTALL_REQUIRES.append(f"ik_solver @ file://{WHEEL_PATH}")

# Installation operation
setup(
    name="GenieSim",
    author="Genie Sim Team",
    maintainer="Genie Sim Team",
    url="https://github.com/AgibotTech/genie_sim",
    version="2.2.2",
    description="geniesim",
    keywords=["agibot", "genie", "sim", "benchmark"],
    license="Mozilla Public License Version 2.0",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    packages=["geniesim"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
