cmake_minimum_required(VERSION 3.8)
project(bio_ik)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_kdl REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenMP)

set(DEPS
  rclcpp
  moveit_core
  moveit_ros_planning
  pluginlib
  tf2_eigen
  tf2_kdl
  geometry_msgs
  moveit_msgs
  kdl_parser
  ament_index_cpp
)

if(OpenMP_CXX_FOUND)
  set(OpenMP_LIBS OpenMP::OpenMP_CXX)
else()
  set(OpenMP_LIBS)
endif()

add_compile_options(-w)
add_compile_options($<$<CONFIG:Release>:-O3>)
add_compile_options($<$<CONFIG:Release>:-ftree-vectorize>)
add_compile_options($<$<CONFIG:Release>:-ffast-math>)
add_compile_options($<$<CONFIG:Release>:-fno-finite-math-only>)

add_library(${PROJECT_NAME} SHARED
  src/goal_types.cpp
  src/problem.cpp
  src/kinematics_plugin.cpp
  src/ik_test.cpp
  src/ik_gradient.cpp
  src/ik_evolution_1.cpp
  src/ik_evolution_2.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Jazzy renamed moveit's public headers from .h to .hpp and emits a
# #pragma warning on every .h include. Detect ROS_DISTRO and propagate
# the switch to anyone consuming our public headers (which transitively
# include moveit_core headers).
set(_ros_distro "$ENV{ROS_DISTRO}")
if(_ros_distro STREQUAL "jazzy" OR
   _ros_distro STREQUAL "kilted" OR
   _ros_distro STREQUAL "rolling")
  target_compile_definitions(${PROJECT_NAME} PUBLIC BIO_IK_MOVEIT_USE_NEW_API)
endif()
target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  orocos-kdl
  yaml-cpp
  ${OpenMP_LIBS}
)
ament_target_dependencies(${PROJECT_NAME} ${DEPS})

pluginlib_export_plugin_description_file(moveit_core bio_ik_kinematics_description.xml)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(${DEPS})

ament_package()
