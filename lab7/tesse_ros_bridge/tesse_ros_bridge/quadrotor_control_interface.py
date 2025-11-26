#!/usr/bin/env python3

import numpy as np
import copy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import tf2_ros
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import Imu, CameraInfo
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    Point,
    PointStamped,
    TransformStamped,
    Quaternion,
    Twist,
    PoseWithCovarianceStamped,
    Vector3Stamped,
)
from ackermann_msgs.msg import AckermannDriveStamped
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge, CvBridgeError

import tesse_ros_bridge.utils
from tesse_ros_bridge.noise_simulator import NoiseParams, NoiseSimulator
from tesse_msgs.srv import (
    SceneRequestService,
    ObjectSpawnRequestService,
    RepositionRequestService,
)
from tesse_msgs.msg import CollisionStats
from tesse_ros_bridge.consts import *
from tesse.msgs import *
from tesse.env import *
from tesse.utils import *


class TesseQuadrotorControlInterface(Node):
    def __init__(self):
        """This class provides a ROS interface for controlling TESSE quadrotor agents.
        
        ROS users can simply send propeller speeds command to predefined propeller speeds
        topic, and this interface will transmit the appropriate messages to TESSE simulator.
        Quadrotor agent is controlled by setting the speed of each propeller. The speeds 
        are radians per second.
        """
        super().__init__('quadrotor_control_interface')
        
        # Declare all parameters
        self._declare_parameters()
        
        # Networking parameters
        self.sim_ip = self.get_parameter('sim_ip').value
        self.self_ip = self.get_parameter('self_ip').value
        self.use_broadcast = self.get_parameter('use_broadcast').value
        self.position_port = self.get_parameter('position_port').value
        self.metadata_port = self.get_parameter('metadata_port').value
        self.image_port = self.get_parameter('image_port').value
        self.udp_port = self.get_parameter('udp_port').value
        self.step_port = self.get_parameter('step_port').value
        self.scan_port = self.get_parameter('lidar_port').value
        self.scan_udp_port = self.get_parameter('lidar_udp_port').value
        
        # Topics
        self.props_speeds_topic = "rotor_speed_cmds"
        
        # Initialize the Env object to communicate with simulator
        self.env = Env(
            simulation_ip=self.sim_ip,
            own_ip=self.self_ip,
            position_port=self.position_port,
            metadata_port=self.metadata_port,
            image_port=self.image_port,
            step_port=self.step_port,
        )
        
        # QoS profile for subscriber
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Setup control interface subscriber
        self.props_speeds_sub = self.create_subscription(
            Actuators,
            self.props_speeds_topic,
            self.props_control_cb,
            qos_profile
        )
        
        self.get_logger().info("Quadrotor control interface initialized")
        self.get_logger().info(f"Listening for rotor speed commands on topic: {self.props_speeds_topic}")
    
    def _declare_parameters(self):
        """Declare all ROS2 parameters with default values"""
        # Networking parameters
        self.declare_parameter('sim_ip', '127.0.0.1')
        self.declare_parameter('self_ip', '127.0.0.1')
        self.declare_parameter('use_broadcast', False)
        self.declare_parameter('position_port', 9000)
        self.declare_parameter('metadata_port', 9001)
        self.declare_parameter('image_port', 9002)
        self.declare_parameter('udp_port', 9004)
        self.declare_parameter('step_port', 9005)
        self.declare_parameter('lidar_port', 9006)
        self.declare_parameter('lidar_udp_port', 9007)
    
    def props_control_cb(self, msg):
        """Callback function used for propeller speed control
        
        Args:
            msg: A mav_msgs.msg.Actuators message. The field angular_velocities 
                 is used for setting the propeller speeds.
        """
        # Read prop speeds
        speeds = msg.angular_velocities
        
        # Check if we have at least 4 propeller speeds
        if len(speeds) < 4:
            self.get_logger().warn(
                f"Received {len(speeds)} propeller speeds, expected 4. Padding with zeros."
            )
            # Pad with zeros if necessary
            speeds = list(speeds) + [0.0] * (4 - len(speeds))
        
        # Send propeller speeds to TESSE simulator
        try:
            self.env.send(PropSpeeds(speeds[0], speeds[1], speeds[2], speeds[3]))
            self.get_logger().debug(
                f"Sent propeller speeds: [{speeds[0]:.2f}, {speeds[1]:.2f}, "
                f"{speeds[2]:.2f}, {speeds[3]:.2f}]"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to send propeller speeds: {e}")


def main(args=None):
    """Main function to initialize and run the quadrotor control interface"""
    rclpy.init(args=args)
    
    node = TesseQuadrotorControlInterface()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down quadrotor control interface...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
