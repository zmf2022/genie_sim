cmake_minimum_required(VERSION 3.16)
project(genie_sim_engine CXX)

if(NOT UNIX OR APPLE)
  message(FATAL_ERROR "genie_sim_engine is Linux-only")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(image_transport REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11_vendor QUIET)
find_package(pybind11 REQUIRED)

pybind11_add_module(genie_sim_engine_py MODULE
  src/pybinding.cpp
  src/realtime_ros_node.cpp
  src/realtime_scheduler.cpp
  src/stats.cpp
  src/realtime_buffer.cpp
)

target_include_directories(genie_sim_engine_py PRIVATE src)

ament_target_dependencies(genie_sim_engine_py PUBLIC
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_msgs
  tf2_ros
  rosgraph_msgs
  std_msgs
  builtin_interfaces
  realtime_tools
  image_transport
)

target_compile_options(genie_sim_engine_py PRIVATE
  -fvisibility=hidden
  -fvisibility-inlines-hidden
  -ffunction-sections
  -fdata-sections
  -fno-math-errno
  -Wall
  -Wextra
  -Wno-unused-parameter
)

# Obfuscation / stripping only on Release builds.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(genie_sim_engine_py PRIVATE -O3 -fno-ident)
  target_link_options(genie_sim_engine_py PRIVATE
    -Wl,--gc-sections
    -Wl,--build-id=none
    -Wl,--strip-all
    -s
  )
endif()

install(TARGETS genie_sim_engine_py
  LIBRARY DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/genie_sim_engine_isaacsim.py scripts/genie_sim_engine_newton.py scripts/assemble_scene.py scripts/assemble_robot.py scripts/assemble_newton.py
  DESTINATION lib/${PROJECT_NAME}
)

# Library modules — installed as a tree under
# lib/${PROJECT_NAME}/{engine,runtime,common,kit}/ so the in-process
# imports (e.g. ``from engine.base import PhysicsEngine``,
# ``from kit.stage import IsaacSimStage``) resolve from
# the PROGRAMS' working directory.
#
# Per-file ``install(FILES ...)`` (not ``install(DIRECTORY ...)``) so
# colcon's ``--symlink-install`` actually symlinks each .py. With
# ``install(DIRECTORY)``, colcon-cmake copies files into the devel tree
# even under symlink-install — meaning a source edit doesn't propagate
# without a rebuild. CONFIGURE_DEPENDS re-globs on every build so newly
# added .py files get picked up; the trade-off is a glob on every
# configure, which is cheap here (< 100 files).
file(GLOB_RECURSE _module_pys
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/scripts"
  CONFIGURE_DEPENDS
  "scripts/engine/*.py"
  "scripts/runtime/*.py"
  "scripts/common/*.py"
  "scripts/kit/*.py"
)
foreach(_rel IN LISTS _module_pys)
  get_filename_component(_subdir "${_rel}" DIRECTORY)
  install(FILES "scripts/${_rel}" DESTINATION "lib/${PROJECT_NAME}/${_subdir}")
endforeach()

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
