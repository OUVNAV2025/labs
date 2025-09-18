cmake_minimum_required(VERSION 3.16)
project(lab_3)

add_compile_options(-std=c++17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
)

# Feature trackers library
add_library(feature_trackers src/feature_tracker.cpp
                             src/sift_feature_tracker.cpp
                             src/akaze_feature_tracker.cpp
                             src/orb_feature_tracker.cpp
                             src/brisk_feature_tracker.cpp
                             src/lk_feature_tracker.cpp)

ament_target_dependencies(feature_trackers OpenCV)

target_include_directories(feature_trackers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Track features executable
add_executable(track_features src/track_features.cpp)
ament_target_dependencies(track_features OpenCV cv_bridge image_transport sensor_msgs)
target_link_libraries(track_features feature_trackers glog::glog ${OpenCV_LIBRARIES})

target_include_directories(track_features PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Self flow executable
add_executable(self_flow src/self_flow.cpp)
ament_target_dependencies(self_flow OpenCV cv_bridge image_transport sensor_msgs)
target_link_libraries(self_flow feature_trackers ${OpenCV_LIBRARIES})

target_include_directories(self_flow PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Installation rules
install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS feature_trackers
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

install(TARGETS track_features self_flow DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY images DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY bags DESTINATION share/${PROJECT_NAME})

ament_package()

