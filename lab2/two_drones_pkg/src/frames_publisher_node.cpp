#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>

class FramesPublisherNode : public rclcpp::Node {
  rclcpp::Time startup_time;

  // Declare a unique pointer for the transform broadcaster
  std::unique_ptr<tf2_ros::TransformBroadcaster> transform_broadcaster;

  rclcpp::TimerBase::SharedPtr heartbeat;

 public:
  FramesPublisherNode() : Node("frames_publisher_node") {
    // Instantiate the TransformBroadcaster
    transform_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    startup_time = this->now();
    
    // Use create_wall_timer to schedule the onPublish callback every 20ms (50Hz)
    heartbeat = this->create_wall_timer(
        std::chrono::milliseconds(20), 
        std::bind(&FramesPublisherNode::onPublish, this)
    );
  }

  void onPublish() {
    // Compute time elapsed in seconds since startup_time
    rclcpp::Duration elapsed_time = this->now() - startup_time;
    double time = elapsed_time.seconds();

    // Declare two transform messages for the AVs
    geometry_msgs::msg::TransformStamped world_T_av1;
    geometry_msgs::msg::TransformStamped world_T_av2;

    // Populate the transforms for the AVs using the calculated time
    // world_T_av1: Position (cos(time), sin(time), 0.0), rotation with y-axis tangent to the trajectory
    world_T_av1.header.stamp = this->now();
    world_T_av1.header.frame_id = "world";
    world_T_av1.child_frame_id = "av1";
    world_T_av1.transform.translation.x = cos(time);
    world_T_av1.transform.translation.y = sin(time);
    world_T_av1.transform.translation.z = 0.0;
    
    tf2::Quaternion q1;
    q1.setRPY(0, time, 0);  // Ensure the y-axis stays tangent to the trajectory
    world_T_av1.transform.rotation.x = q1.x();
    world_T_av1.transform.rotation.y = q1.y();
    world_T_av1.transform.rotation.z = q1.z();
    world_T_av1.transform.rotation.w = q1.w();

    // world_T_av2: Position (sin(time), 0.0, cos(2*time)), with irrelevant rotation
    world_T_av2.header.stamp = this->now();
    world_T_av2.header.frame_id = "world";
    world_T_av2.child_frame_id = "av2";
    world_T_av2.transform.translation.x = sin(time);
    world_T_av2.transform.translation.y = 0.0;
    world_T_av2.transform.translation.z = cos(2 * time);

    // Send the transforms
    transform_broadcaster->sendTransform(world_T_av1);
    transform_broadcaster->sendTransform(world_T_av2);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FramesPublisherNode>());
  rclcpp::shutdown();
  return 0;
}

