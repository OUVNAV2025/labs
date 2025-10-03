cmake_minimum_required(VERSION 3.16)
project(lab_4)

add_compile_options(-std=c++17)

# Required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(lab_3 REQUIRED)
find_package(opengv REQUIRED)

# Executable
add_executable(pose_estimation src/pose_estimation.cpp)

# Dependencies
ament_target_dependencies(pose_estimation
  rclcpp
  geometry_msgs
  nav_msgs
  image_transport
  cv_bridge
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  lab_3
  opengv
)

# Libraries to link manually
target_link_libraries(pose_estimation
  glog::glog
  opengv
)

# Include directories
target_include_directories(pose_estimation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Install targets and resources
install(TARGETS pose_estimation
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY bags
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
