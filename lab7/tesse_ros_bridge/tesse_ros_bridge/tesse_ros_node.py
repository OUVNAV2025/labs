#!/usr/bin/env python3

import numpy as np
import copy
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import tf2_ros
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, TransformListener, Buffer

from std_msgs.msg import Header, String
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import Imu, CameraInfo
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseStamped, Point, \
     PointStamped, TransformStamped, Quaternion, \
     Twist, PoseWithCovarianceStamped, Vector3Stamped
from ackermann_msgs.msg import AckermannDriveStamped
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge, CvBridgeError
from builtin_interfaces.msg import Time as TimeMsg

import tesse_ros_bridge.utils
from tesse_ros_bridge.noise_simulator import NoiseParams, NoiseSimulator

from tesse_msgs.srv import SceneRequestService, \
     ObjectSpawnRequestService, RepositionRequestService
from tesse_msgs.msg import CollisionStats
from tesse_ros_bridge.consts import *

from tesse.msgs import *
from tesse.env import *
from tesse.utils import *

import tf_transformations


def from_sec_float(seconds):
    """Convert float seconds to ROS2 Time message"""
    secs = int(seconds)
    nsecs = int((seconds - secs) * 1e9)
    return TimeMsg(sec=secs, nanosec=nsecs)


def to_sec_float(time_msg):
    """Convert ROS2 Time message to float seconds"""
    return float(time_msg.sec) + float(time_msg.nanosec) * 1e-9


def mk_transform(trans, quat, timestamp, child_frame_id, frame_id):
    """Create a TransformStamped message"""
    t = TransformStamped()
    t.header.stamp = timestamp
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id

    t.transform.translation.x = trans[0]
    t.transform.translation.y = trans[1]
    t.transform.translation.z = trans[2]

    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]

    return t


class TesseROSWrapper(Node):

    def __init__(self):
        super().__init__('tesse_ros_node')
        
        # Declare all parameters
        self._declare_parameters()
        
        # Interface parameters
        self.step_mode_enabled = self.get_parameter('enable_step_mode').value

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

        # Set data to publish
        self.publish_clock = self.get_parameter('publish_clock').value
        self.publish_metadata = self.get_parameter('publish_metadata').value
        self.publish_collisions = self.get_parameter('publish_collisions').value
        self.publish_imu = self.get_parameter('publish_imu').value
        self.publish_odom = self.get_parameter('publish_odom').value
        self.publish_noisy_imu = self.get_parameter('publish_noisy_imu').value
        self.publish_imu_noise_biases = self.get_parameter('publish_imu_noise_biases').value
        self.publish_noisy_odom = self.get_parameter('publish_noisy_odom').value
        self.publish_stereo_rgb = self.get_parameter('publish_stereo_rgb').value
        self.publish_stereo_gry = self.get_parameter('publish_stereo_gry').value
        self.publish_segmentation = self.get_parameter('publish_segmentation').value
        self.publish_depth = self.get_parameter('publish_depth').value
        self.publish_third_pov = self.get_parameter('publish_third_pov').value
        self.publish_front_lidar = self.get_parameter('publish_front_lidar').value
        self.publish_rear_lidar = self.get_parameter('publish_rear_lidar').value

        # Simulator speed parameters
        self.frame_rate = self.get_parameter('frame_rate').value
        self.imu_rate = self.get_parameter('imu_rate').value
        self.scan_rate = self.get_parameter('scan_rate').value

        # Output parameters
        self.use_gt_frames = self.get_parameter('use_gt_frames').value
        self.world_frame_id = self.get_parameter('world_frame_id').value
        self.body_frame_id = self.get_parameter('body_frame_id').value
        self.body_frame_id_gt = self.get_parameter('body_frame_id_gt').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        
        # Init noisification parameters
        self.noise_params = NoiseParams()

        # Init noise simulator
        self.noise_simulator = NoiseSimulator(self.noise_params)

        self.imu_gyro_bias_vec = Vector3Stamped()
        self.imu_accel_bias_vec = Vector3Stamped()
        if self.use_gt_frames:
            self.imu_gyro_bias_vec.header.frame_id = self.body_frame_id_gt
            self.imu_accel_bias_vec.header.frame_id = self.body_frame_id_gt
        else:
            self.imu_gyro_bias_vec.header.frame_id = self.body_frame_id
            self.imu_accel_bias_vec.header.frame_id = self.body_frame_id

        # Initialize the Env object to communicate with simulator
        self.env = Env(simulation_ip=self.sim_ip,
                       own_ip=self.self_ip,
                       position_port=self.position_port,
                       metadata_port=self.metadata_port,
                       image_port=self.image_port,
                       step_port=self.step_port)

        # Setup ROS services
        self.setup_ros_services()

        # Setup simulator step mode with teleop
        if self.step_mode_enabled:
            self.last_step_cmd = []
            self.env.send(SetFrameRate(self.frame_rate))

        # Setup collision
        enable_collision = self.get_parameter('enable_collision').value
        if not enable_collision:
            self.setup_collision(enable_collision)

        # Change scene
        initial_scene = self.get_parameter('initial_scene').value
        # Note: Scene change service needs to be implemented/tested
        # self.change_scene(initial_scene)

        # To send images via ROS network and convert from/to ROS
        self.cv_bridge = CvBridge()

        # QoS profile for publishers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Transform broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_tfs_to_broadcast = []
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Setup all sensor data publishers and sensor objects for interfacing
        self.cameras = []
        self.img_pubs = []
        self.cam_info_pubs = []
        self.cam_info_msgs = []
        self.cam_params = []
        self.lidars = []
        self.scan_pubs = []
        self.lidar_params = []

        self.setup_all_cameras()
        self.setup_all_lidars()

        # Publish all sensor static TFs
        if self.static_tfs_to_broadcast:
            self.static_tf_broadcaster.sendTransform(self.static_tfs_to_broadcast)

        # Setup metadata publisher
        if self.publish_metadata:
            self.metadata_pub = self.create_publisher(String, 'metadata', qos_profile)

        # If the clock updates faster than images can be queried in
        # step mode, the image callback is called twice on the same
        # timestamp which leads to duplicate published images.
        # Track image timestamps to prevent this
        self.last_image_timestamp = None

        # Setup ROS publishers for metadata
        if self.publish_imu:
            self.clean_imu_pub = self.create_publisher(Imu, 'imu/clean/imu', qos_profile)

        if self.publish_odom:
            self.odom_pub = self.create_publisher(Odometry, 'odom', qos_profile)

        if self.publish_noisy_imu:
            self.noisy_imu_pub = self.create_publisher(Imu, 'imu/noisy/imu', qos_profile)
            self.imu_gyro_bias_pub = self.create_publisher(
                Vector3Stamped, 'imu/noisy/biases/gyro', qos_profile)
            self.imu_accel_bias_pub = self.create_publisher(
                Vector3Stamped, 'imu/noisy/biases/accel', qos_profile)

        if self.publish_noisy_odom:
            self.noisy_odom_pub = self.create_publisher(Odometry, 'odom/noisy', qos_profile)

        if self.publish_collisions:
            self.coll_pub = self.create_publisher(CollisionStats, 'collision', qos_profile)

        # Required states for finite difference calculations
        self.prev_time = 0.0
        self.prev_vel_brh = [0.0, 0.0, 0.0]
        self.prev_enu_R_brh = np.identity(3)

        # Spawn initial objects
        self.spawn_initial_objects()

        # Setup metadata UdpListener
        udp_host = self.self_ip
        if self.use_broadcast:
            udp_host = '<broadcast>'
        self.meta_listener = UdpListener(host=udp_host,
                                         port=self.udp_port,
                                         rate=self.imu_rate)
        self.meta_listener.subscribe('udp_subscriber', self.meta_cb)

        # Simulated time requires that we constantly publish to '/clock'.
        if self.publish_clock:
            self.clock_pub = self.create_publisher(Clock, '/clock', qos_profile)

        # Setup initial-pose subscriber
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.reposition_cb,
            qos_profile)

        # Setup driving commands
        ackermann_drive = self.get_parameter('drive_with_ackermann').value

        if ackermann_drive:
            self.create_subscription(
                AckermannDriveStamped,
                'drive',
                self.cmd_cb_ackermann,
                qos_profile)
        else:
            self.create_subscription(
                Twist,
                'drive',
                self.cmd_cb_twist,
                qos_profile)

        self.get_logger().info("TESSE_ROS_NODE: Initialization complete.")

    def _declare_parameters(self):
        """Declare all ROS2 parameters with default values"""
        # Interface parameters
        self.declare_parameter('enable_step_mode', False)
        
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
        
        # Publish flags
        self.declare_parameter('publish_clock', False)
        self.declare_parameter('publish_metadata', False)
        self.declare_parameter('publish_collisions', False)
        self.declare_parameter('publish_imu', False)
        self.declare_parameter('publish_odom', False)
        self.declare_parameter('publish_noisy_imu', False)
        self.declare_parameter('publish_imu_noise_biases', False)
        self.declare_parameter('publish_noisy_odom', False)
        self.declare_parameter('publish_stereo_rgb', False)
        self.declare_parameter('publish_stereo_gry', False)
        self.declare_parameter('publish_segmentation', False)
        self.declare_parameter('publish_depth', False)
        self.declare_parameter('publish_third_pov', False)
        self.declare_parameter('publish_front_lidar', False)
        self.declare_parameter('publish_rear_lidar', False)
        
        # Rate parameters
        self.declare_parameter('frame_rate', 20.0)
        self.declare_parameter('imu_rate', 200.0)
        self.declare_parameter('scan_rate', 200.0)
        
        # Frame parameters
        self.declare_parameter('use_gt_frames', False)
        self.declare_parameter('world_frame_id', 'world')
        self.declare_parameter('body_frame_id', 'base_link')
        self.declare_parameter('body_frame_id_gt', 'base_link_gt')
        self.declare_parameter('map_frame_id', 'map')
        
        # Other parameters
        self.declare_parameter('enable_collision', True)
        self.declare_parameter('initial_scene', 1)
        self.declare_parameter('drive_with_ackermann', False)
        self.declare_parameter('num_objects', 0)
        self.declare_parameter('camera_params', rclpy.Parameter.Type.STRING)
        self.declare_parameter('lidar_params', rclpy.Parameter.Type.STRING)

    def spin_node(self):
        """Start timers and callbacks"""
        self.meta_listener.start()

        num_objects = self.get_parameter('num_objects').value
        if num_objects > 0:
            self.create_timer(1.0 / self.frame_rate, self.object_cb)

        if len(self.cameras) > 0:
            self.create_timer(1.0 / self.frame_rate, self.image_cb)

        if len(self.lidars) > 0:
            self.create_timer(1.0 / self.scan_rate, self.scan_cb_slow)

        # Create clock timer if needed
        if self.publish_clock:
            self.create_timer(1.0 / self.frame_rate, self.clock_cb)

        rclpy.spin(self)

    def clock_cb(self, timer=None):
        """Publishes simulated clock time as well as collision statistics"""
        if self.step_mode_enabled:
            if len(self.last_step_cmd) > 0:
                cur_cmd = self.last_step_cmd.pop(0)
                self.env.send(StepWithForce(force_z=cur_cmd[0],
                                            torque_y=cur_cmd[2],
                                            force_x=cur_cmd[1]))
            else:
                self.get_logger().debug("No commands to publish...")

        if self.publish_clock or self.publish_collisions:
            try:
                sim_data = self.env.request(MetadataRequest()).metadata
                metadata = tesse_ros_bridge.utils.parse_metadata(sim_data)

                if self.publish_clock:
                    curr_ros_time = from_sec_float(metadata['time'])
                    c = Clock()
                    c.clock = curr_ros_time
                    self.clock_pub.publish(c)

                # Publish collision statistics if necessary
                if self.publish_collisions and metadata['collision_status']:
                    coll_msg = CollisionStats()
                    coll_msg.header.frame_id = self.body_frame_id_gt
                    coll_msg.header.stamp = curr_ros_time
                    coll_msg.is_collision = metadata['collision_status']
                    coll_msg.object_name = metadata['collision_object']
                    self.coll_pub.publish(coll_msg)

            except Exception as error:
                self.get_logger().error(f"clock_cb error: {error}")

    def cmd_cb_ackermann(self, msg):
        """Listens to published drive commands and sends to simulator"""
        turn_angle_deg = np.rad2deg(msg.drive.steering_angle)
        speed = msg.drive.speed

        # Messages are in SI, RCC requires KMH:
        speed_kmh = speed * 3.6

        self.env.send(SetSpeed(np.abs(speed_kmh)))
        self.env.send(SetTurnSpeed(turn_angle_deg))

        # Move forward/backwards
        if speed_kmh > 0:
            self.env.send(Drive(1))
        elif speed_kmh < 0:
            self.env.send(Drive(-1))

        # Turn left/right
        if turn_angle_deg > 0:
            self.env.send(Turn(1))
        elif turn_angle_deg < 0:
            self.env.send(Turn(-1))

    def cmd_cb_twist(self, msg):
        """Listens to teleop force commands and sends to simulator"""
        force_x = msg.linear.x
        force_y = msg.linear.y
        torque_z = msg.angular.z

        if self.step_mode_enabled:
            self.last_step_cmd.append([force_x, force_y, torque_z])
        else:
            self.env.send(AddForce(force_z=force_x,
                                   torque_y=torque_z,
                                   force_x=force_y))

    def reposition_cb(self, msg):
        """Listens to pose requests and sends the reposition command to the simulator"""
        pose = msg.pose.pose
        if msg.header.frame_id != self.world_frame_id:
            pst = PoseStamped()
            pst.pose = msg.pose.pose
            pst.header = msg.header
            
            try:
                # Transform pose to world frame
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame_id,
                    msg.header.frame_id,
                    Time())
                # Apply transformation (simplified, may need tf2_geometry_msgs)
                pose = pst.pose
            except Exception as e:
                self.get_logger().error(f"Transform lookup failed: {e}")
                return

        # Hacky frame change from ROS to Unity (left handed)
        py = pose.position.y
        pz = pose.position.z
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w

        # Rotate 90 degrees around y in the most lazy way possible...
        quat = tf_transformations.quaternion_multiply([-qx, -qz, -qy, qw],
                                                      [0, 0.7071068, 0, 0.7071068])

        pose.position.y = -pz
        pose.position.z = py
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        self.reposition_agent(pose)

    def meta_cb(self, data):
        """Callback for UDP metadata at high rates"""
        # Publish raw metadata
        if self.publish_metadata:
            msg = String()
            msg.data = data if isinstance(data, str) else data.decode()
            self.metadata_pub.publish(msg)

        # Parse metadata and process for proper use
        metadata = tesse_ros_bridge.utils.parse_metadata(data)
        assert(self.prev_time < metadata['time'])

        metadata_processed = tesse_ros_bridge.utils.process_metadata(metadata,
                self.prev_time, self.prev_vel_brh, self.prev_enu_R_brh)

        timestamp = from_sec_float(metadata_processed['time'])

        # Publish agent ground truth transform
        self.publish_tf(metadata_processed['transform'], timestamp)

        # Publish clean imu and odometry messages
        if self.publish_imu:
            if self.use_gt_frames:
                imu = tesse_ros_bridge.utils.metadata_to_imu(metadata_processed,
                        timestamp, self.body_frame_id_gt)
            else:
                imu = tesse_ros_bridge.utils.metadata_to_imu(metadata_processed,
                        timestamp, self.body_frame_id)
            self.clean_imu_pub.publish(imu)

        if self.publish_odom:
            odom = tesse_ros_bridge.utils.metadata_to_odom(metadata_processed,
                    timestamp, self.world_frame_id, self.body_frame_id_gt)
            self.odom_pub.publish(odom)

        # Publish noisy imu and odometry messages
        if self.publish_noisy_imu or self.publish_noisy_odom:
            metadata_noisy = self.noise_simulator.apply_noise_to_metadata(metadata_processed)

            if self.publish_noisy_imu:
                if self.use_gt_frames:
                    imu = tesse_ros_bridge.utils.metadata_to_imu(metadata_noisy,
                            timestamp, self.body_frame_id_gt)
                else:
                    imu = tesse_ros_bridge.utils.metadata_to_imu(metadata_noisy,
                            timestamp, self.body_frame_id)
                self.noisy_imu_pub.publish(imu)

                # Publish imu biases for debugging
                if self.publish_imu_noise_biases:
                    self.imu_gyro_bias_vec.header.stamp = timestamp
                    self.imu_gyro_bias_vec.vector.x = self.noise_simulator.gyroscope_bias[0]
                    self.imu_gyro_bias_vec.vector.y = self.noise_simulator.gyroscope_bias[1]
                    self.imu_gyro_bias_vec.vector.z = self.noise_simulator.gyroscope_bias[2]
                    self.imu_gyro_bias_pub.publish(self.imu_gyro_bias_vec)

                    self.imu_accel_bias_vec.header.stamp = timestamp
                    self.imu_accel_bias_vec.vector.x = self.noise_simulator.accelerometer_bias[0]
                    self.imu_accel_bias_vec.vector.y = self.noise_simulator.accelerometer_bias[1]
                    self.imu_accel_bias_vec.vector.z = self.noise_simulator.accelerometer_bias[2]
                    self.imu_accel_bias_pub.publish(self.imu_accel_bias_vec)

            if self.publish_noisy_odom:
                odom = tesse_ros_bridge.utils.metadata_to_odom(metadata_noisy,
                        timestamp, self.world_frame_id, self.body_frame_id_gt)
                self.noisy_odom_pub.publish(odom)

        self.prev_time = metadata_processed['time']
        self.prev_vel_brh = metadata_processed['velocity']
        self.prev_enu_R_brh = metadata_processed['transform'][:3,:3]

    def image_cb(self, timer=None):
        """Publish images from simulator to ROS"""
        try:
            # Get camera data
            data_response = self.env.request(DataRequest(True, self.cameras))

            # Process metadata to publish transform
            metadata = tesse_ros_bridge.utils.parse_metadata(data_response.metadata)
            timestamp = from_sec_float(metadata['time'])

            if timestamp == self.last_image_timestamp:
                self.get_logger().info(f"Skipping duplicate images at timestamp {self.last_image_timestamp}")
                return

            # Process each image
            for i in range(len(self.cameras)):
                if self.cameras[i][0] == Camera.DEPTH:
                    far_draw_dist = self.cam_params[i]['draw_distance']['far']
                    img_msg = self.cv_bridge.cv2_to_imgmsg(
                        data_response.images[i] * far_draw_dist,
                            'passthrough')
                elif self.cameras[i][2] == Channels.SINGLE:
                    img_msg = self.cv_bridge.cv2_to_imgmsg(
                        data_response.images[i], 'mono8')
                elif self.cameras[i][2] == Channels.THREE:
                    img_msg = self.cv_bridge.cv2_to_imgmsg(
                        data_response.images[i], 'rgb8')

                # Sanity check resolutions
                if self.cam_info_msgs[i] and self.cameras[i][0] != Camera.THIRD_PERSON:
                    assert(img_msg.width == self.cam_info_msgs[i].width)
                    assert(img_msg.height == self.cam_info_msgs[i].height)

                # Publish images to appropriate topic
                img_msg.header.frame_id = self.cameras[i][3]
                img_msg.header.stamp = timestamp
                self.img_pubs[i].publish(img_msg)

                # Publish associated CameraInfo message
                if self.cam_info_msgs[i] and self.cam_info_pubs[i]:
                    self.cam_info_msgs[i].header.stamp = timestamp
                    self.cam_info_pubs[i].publish(self.cam_info_msgs[i])

            self.publish_tf(
                tesse_ros_bridge.utils.get_enu_T_brh(metadata),
                    timestamp)

            self.last_image_timestamp = timestamp

        except Exception as error:
            self.get_logger().error(f"image_cb error: {error}")

    def object_cb(self, timer=None):
        """Publish object transforms"""
        try:
            obj_metadata = self.env.request(ObjectsRequest()).metadata
            obj_dict = tesse_ros_bridge.utils.parse_object_data(obj_metadata)

            for id, obj in obj_dict.items():
                assert(type(obj) == dict)
                assert(type(id) == int)

                frame_id = "object_" + str(id)
                self.tf_broadcaster.sendTransform(mk_transform(
                                                  obj['position'],
                                                  obj['quaternion'],
                                                  from_sec_float(obj['time']),
                                                  frame_id,
                                                  self.world_frame_id))
        except Exception as error:
            self.get_logger().error(f"object_cb error: {error}")

    def scan_cb_slow(self, timer=None):
        """Received LiDAR data from the simulator using standard udp requests"""
        try:
            data_response = self.env.request(
                LidarDataRequest(True, [lidar[0] for lidar in self.lidars]))
            self.scan_cb(data_response)
        except Exception as error:
            self.get_logger().error(f"scan_cb_slow error: {error}")

    def scan_cb(self, data):
        """Receives LiDAR data from the simulator and publishes to ROS"""
        # Parse metadata and process for proper use
        metadata = tesse_ros_bridge.utils.parse_metadata(data.metadata)
        timestamp = from_sec_float(metadata['time'])

        # Publish scan messages
        scan = LaserScan()
        scan.header.stamp = timestamp
        for i in range(len(self.lidars)):
            scan.header.frame_id = self.lidars[i][1]

            # TODO: this is a hack! for some reason the lidar scans are aligned
            # with the y-axis in this frame. You must "transform" the scans!
            scan.angle_min = self.lidar_params[i]['parameters']['min_angle'] - np.pi/2.0
            scan.angle_max = self.lidar_params[i]['parameters']['max_angle'] - np.pi/2.0
            scan.angle_increment = self.lidar_params[i]['parameters']['angle_inc']
            scan.scan_time = 1.0 / self.scan_rate
            scan.range_min = 0.0
            scan.range_max = self.lidar_params[i]['parameters']['max_range']
            scan.ranges = data.scans[i]

            self.scan_pubs[i].publish(scan)

    def setup_camera(self, camera_params):
        """Set the parameters, position, and orientation of one camera"""
        # Determine Unity camera, compression, channels
        camera_type_switcher = {
            "left_cam": Camera.RGB_LEFT,
            "right_cam": Camera.RGB_RIGHT,
            "seg_cam": Camera.SEGMENTATION,
            "depth_cam": Camera.DEPTH,
            "third_person": Camera.THIRD_PERSON,
        }
        camera_id = camera_type_switcher[camera_params['camera_id']]

        n_channel_switcher = {
            1: Channels.SINGLE,
            3: Channels.THREE,
        }
        n_channels = n_channel_switcher[camera_params['num_channels']]

        # Depth camera is a special case
        if camera_id == Camera.DEPTH:
            n_channels = Channels.THREE

        # Uniqueness of camera_id is necessary to prevent extra camera setup
        unique_camera_id = True
        for camera in self.cameras:
            if camera[0] == camera_id:
                unique_camera_id = False

        # Store camera object for callbacks
        camera = (camera_id,
                  Compression.ON if camera_params['compression'] else Compression.OFF,
                  n_channels,
                  camera_params['frame_id'],
        )
        self.cameras.append(camera)

        if unique_camera_id:
            width = camera_params['width']
            height = camera_params['height']
            vertical_fov = camera_params['vertical_fov']
            near_draw_dist = camera_params['near_draw_dist']
            far_draw_dist = camera_params['far_draw_dist']

            pos_x = camera_params['pos_x']
            pos_y = camera_params['pos_y']
            pos_z = camera_params['pos_z']

            quat_x = camera_params['quat_x']
            quat_y = camera_params['quat_y']
            quat_z = camera_params['quat_z']
            quat_w = camera_params['quat_w']

            # Set parameters
            resp = None
            while resp is None:
                self.get_logger().info(f"Setting intrinsic parameters for camera: {camera_id}")
                resp = self.env.request(SetCameraParametersRequest(
                    camera_id,
                    height,
                    width,
                    vertical_fov,
                    near_draw_dist,
                    far_draw_dist))

            # Set position
            resp = None
            while resp is None:
                self.get_logger().info(f"Setting position of camera: {camera_id}")
                resp = self.env.request(SetCameraPositionRequest(
                        camera_id,
                        pos_x,
                        pos_y,
                        pos_z,))

            # Set orientation
            while resp is None:
                self.get_logger().info(f"Setting orientation of camera: {camera_id}")
                resp = self.env.request(SetCameraOrientationRequest(
                        camera_id,
                        quat_x,
                        quat_y,
                        quat_z,
                        quat_w,))

            # Get information back from simulator
            cam_data = None
            while cam_data is None:
                self.get_logger().info(f"Acquiring camera data for camera: {camera_id}")
                cam_data = tesse_ros_bridge.utils.parse_cam_data(
                    self.env.request(
                        CameraInformationRequest(camera_id)).metadata)

            assert(cam_data['id'] == camera_id.value)
            assert(cam_data['parameters']['height'] == height)
            assert(cam_data['parameters']['width'] == width)
            self.cam_params.append(cam_data)

            # Store static transform for camera
            static_tf_cam = TransformStamped()
            static_tf_cam.header.frame_id = self.body_frame_id
            if self.use_gt_frames:
                static_tf_cam.header.frame_id = self.body_frame_id_gt
            static_tf_cam.child_frame_id = camera[3]
            static_tf_cam.header.stamp = self.get_clock().now().to_msg()
            static_tf_cam.transform.translation = Point(
                x=cam_data['position'][0],
                y=cam_data['position'][1],
                z=cam_data['position'][2])
            static_tf_cam.transform.rotation = Quaternion(
                x=cam_data['quaternion'][0],
                y=cam_data['quaternion'][1],
                z=cam_data['quaternion'][2],
                w=cam_data['quaternion'][3])

            self.static_tfs_to_broadcast.append(static_tf_cam)

            # Store camera information message for publishing
            cam_info_msg = tesse_ros_bridge.utils.generate_camera_info(
                    cam_data, camera[3])

            # Initialize the publisher for the camera info
            cam_info_pub = self.create_publisher(
                CameraInfo,
                camera_params['camera_id'] + "/camera_info",
                10)

            self.cam_info_msgs.append(cam_info_msg)
            self.cam_info_pubs.append(cam_info_pub)

        else:
            # We have to append something to keep lengths correct
            self.cam_params.append(None)
            self.cam_info_msgs.append(None)
            self.cam_info_pubs.append(None)

        # Initialize the publisher for the image
        if n_channel_switcher[camera_params['num_channels']] == Channels.THREE:
            self.img_pubs.append(self.create_publisher(
                ImageMsg,
                camera_params['camera_id'] + "/rgb/image_raw",
                10))
        elif n_channel_switcher[camera_params['num_channels']] == Channels.SINGLE:
            self.img_pubs.append(self.create_publisher(
                ImageMsg,
                camera_params['camera_id'] + "/mono/image_raw",
                10))

    def setup_all_cameras(self):
        """Sets up all cameras based on the publish flags passed from launch"""
        try:
            camera_params = self.get_parameter('camera_params').value
            if not camera_params or camera_params == 'NOT SET':
                camera_params = {}
        except:
            camera_params = {}

        if self.publish_stereo_rgb:
            if 'RGB_LEFT' in camera_params:
                self.setup_camera(camera_params['RGB_LEFT'])
            if 'RGB_RIGHT' in camera_params:
                self.setup_camera(camera_params['RGB_RIGHT'])

        if self.publish_stereo_gry:
            if 'GRY_LEFT' in camera_params:
                self.setup_camera(camera_params['GRY_LEFT'])
            if 'GRY_RIGHT' in camera_params:
                self.setup_camera(camera_params['GRY_RIGHT'])

        if self.publish_segmentation:
            if 'SEGMENTATION' in camera_params:
                self.setup_camera(camera_params['SEGMENTATION'])

        if self.publish_depth:
            if 'DEPTH' in camera_params:
                self.setup_camera(camera_params['DEPTH'])

        if self.publish_third_pov:
            if 'THIRD_PERSON' in camera_params:
                self.setup_camera(camera_params['THIRD_PERSON'])

    def setup_lidar(self, lidar_params):
        """Set up one LiDAR in the simulator"""
        lidar_type_switcher = {
            "front_lidar": Lidar.HOOD,
            "rear_lidar": Lidar.TRUNK,
        }
        lidar_id = lidar_type_switcher[lidar_params['lidar_id']]

        lidar = (lidar_id, lidar_params['frame_id'])
        self.lidars.append(lidar)

        # Get all lidar parameters
        min_angle = lidar_params["scan_min_angle"]
        max_angle = lidar_params["scan_max_angle"]
        max_range = lidar_params["scan_max_range"]
        num_beams = lidar_params["scan_beams"]

        pos_x = lidar_params["pos_x"]
        pos_y = lidar_params["pos_y"]
        pos_z = lidar_params["pos_z"]

        quat_x = lidar_params["quat_x"]
        quat_y = lidar_params["quat_y"]
        quat_z = lidar_params["quat_z"]
        quat_w = lidar_params["quat_w"]

        # Set parameters
        resp = None
        while resp is None:
            self.get_logger().info(f"Setting intrinsic parameters for lidar: {lidar_id}")
            resp = self.env.request(SetLidarParametersRequest(
                lidar_id,
                min_angle=min_angle,
                max_angle=max_angle,
                max_range=max_range,
                ray_count=num_beams))

        # Set position
        if pos_x != "default" or pos_y != "default" or pos_z != "default":
            resp = None
            while resp is None:
                self.get_logger().info(f"Setting position of lidar: {lidar_id}")
                resp = self.env.request(SetLidarPositionRequest(
                        lidar_id,
                        pos_x,
                        pos_y,
                        pos_z))

        # Set orientation
        if quat_x != "default" or quat_y != "default" or \
                quat_z != "default" or quat_w != "default":
            resp = None
            while resp is None:
                self.get_logger().info(f"Setting orientation of lidar: {lidar_id}")
                resp = self.env.request(SetLidarOrientationRequest(
                        lidar_id,
                        quat_x,
                        quat_y,
                        quat_z,
                        quat_w))

        # Get information back from simulator
        lidar_data = None
        while lidar_data is None:
            self.get_logger().info(f"Acquiring lidar data for lidar: {lidar_id}")
            lidar_data = tesse_ros_bridge.utils.parse_lidar_data(
                self.env.request(LidarInformationRequest(lidar_id)).metadata)

            tol = 1e-6
            assert(lidar_data["id"] == lidar_id.value)
            assert(abs(lidar_data['parameters']["min_angle"] - min_angle) < tol)
            assert(abs(lidar_data['parameters']["max_angle"] - max_angle) < tol)
            assert(abs(lidar_data['parameters']["max_range"] - max_range) < tol)
            assert(abs(lidar_data['parameters']["ray_count"] - num_beams) < tol)

        # Publish static transform for lidar
        scan_ts = TransformStamped()
        scan_ts.header.stamp = self.get_clock().now().to_msg()
        scan_ts.header.frame_id = self.body_frame_id
        if self.use_gt_frames:
            scan_ts.header.frame_id = self.body_frame_id_gt
        scan_ts.child_frame_id = lidar[1]
        scan_ts.transform.translation = Point(
            x=lidar_data['position'][0],
            y=lidar_data['position'][1],
            z=lidar_data['position'][2])
        scan_ts.transform.rotation = Quaternion(
            x=lidar_data['quaternion'][0],
            y=lidar_data['quaternion'][1],
            z=lidar_data['quaternion'][2],
            w=lidar_data['quaternion'][3])

        self.static_tfs_to_broadcast.append(scan_ts)
        self.lidar_params.append(lidar_data)

        # Set up scan publisher
        self.scan_pubs.append(
            self.create_publisher(LaserScan, lidar[1] + "/scan", 10))

    def setup_all_lidars(self):
        """Sets up all LiDARs in the simulator"""
        try:
            lidar_params = self.get_parameter('lidar_params').value
            if not lidar_params or lidar_params == 'NOT SET':
                lidar_params = {}
        except:
            lidar_params = {}

        if self.publish_front_lidar:
            if 'FRONT' in lidar_params:
                self.setup_lidar(lidar_params['FRONT'])

        if self.publish_rear_lidar:
            if 'REAR' in lidar_params:
                self.setup_lidar(lidar_params['REAR'])

    def spawn_initial_objects(self):
        """Spawn initial objects from the parameter yaml file"""
        self.get_logger().info("Spawning initial objects")
        num_objects = self.get_parameter('num_objects').value

        for i in range(num_objects):
            try:
                obj_dict = self.get_parameter(f'object_{i}').value
                pose = Pose()
                params = []
                if obj_dict['use_custom_pose']:
                    pose.position.x = obj_dict['px']
                    pose.position.y = obj_dict['py']
                    pose.position.z = obj_dict['pz']
                    pose.orientation.x = obj_dict['qx']
                    pose.orientation.y = obj_dict['qy']
                    pose.orientation.z = obj_dict['qz']
                    pose.orientation.w = obj_dict['qw']
                if obj_dict['send_params']:
                    if 'params' in obj_dict.keys():
                        for param in obj_dict['params']:
                            params.append(param)
                    else:
                        # Choose random parameters
                        params = [(np.random.random() * 2) - 1 for i in range(10)]

                self.spawn_object(obj_dict['id'], pose, params)
            except Exception as e:
                self.get_logger().error(f"Failed to spawn object {i}: {e}")

    def setup_collision(self, enable_collision):
        """Enable/Disable collisions in Simulator"""
        self.get_logger().info(f"Setup collisions to: {enable_collision}")
        if enable_collision is True:
            self.env.send(ColliderRequest(enable=1))
        else:
            self.env.send(ColliderRequest(enable=0))

    def setup_ros_services(self):
        """Setup ROS services related to the simulator"""
        self.scene_request_service = self.create_service(
            SceneRequestService,
            'scene_change_request',
            self.rosservice_change_scene)

        self.object_spawn_service = self.create_service(
            ObjectSpawnRequestService,
            'object_spawn_request',
            self.rosservice_spawn_object)

        self.reposition_request_service = self.create_service(
            RepositionRequestService,
            'reposition_request',
            self.rosservice_reposition)

    def rosservice_change_scene(self, request, response):
        """Change scene ID of simulator as a ROS service"""
        try:
            self.env.request(SceneRequest(request.id))
            response.success = True
        except Exception as e:
            self.get_logger().error(f"Scene Change Error: {e}")
            response.success = False
        return response

    def rosservice_spawn_object(self, request, response):
        """Spawn an object into the simulator as a ROS service"""
        try:
            if request.pose == Pose():
                self.env.request(SpawnObjectRequest(object_index=request.id,
                                                    method=ObjectSpawnMethod.RANDOM,
                                                    params=request.params))
            else:
                self.env.request(SpawnObjectRequest(object_index=request.id,
                                                    method=ObjectSpawnMethod.USER,
                                                    position_x=request.pose.position.x,
                                                    position_y=request.pose.position.y,
                                                    position_z=request.pose.position.z,
                                                    orientation_x=request.pose.orientation.x,
                                                    orientation_y=request.pose.orientation.y,
                                                    orientation_z=request.pose.orientation.z,
                                                    orientation_w=request.pose.orientation.w,
                                                    params=request.params))
            response.success = True
        except Exception as e:
            self.get_logger().error(f"Object Spawn Error: {e}")
            response.success = False
        return response

    def rosservice_reposition(self, request, response):
        """Repositions the agent to a desired pose as a ROS service"""
        try:
            self.env.send(Reposition(request.pose.position.x,
                          request.pose.position.y,
                          request.pose.position.z,
                          request.pose.orientation.x,
                          request.pose.orientation.y,
                          request.pose.orientation.z,
                          request.pose.orientation.w))
            response.success = True
        except Exception as e:
            self.get_logger().error(f"Reposition Error: {e}")
            response.success = False
        return response

    def spawn_object(self, obj_id, pose, params):
        """Helper method to spawn objects"""
        req = ObjectSpawnRequestService.Request()
        req.id = obj_id
        req.pose = pose
        req.params = params
        # Call service (needs proper client setup)
        # This is simplified - in practice you'd use a service client
        return self.rosservice_spawn_object(req, ObjectSpawnRequestService.Response())

    def reposition_agent(self, pose):
        """Helper method to reposition agent"""
        req = RepositionRequestService.Request()
        req.pose = pose
        return self.rosservice_reposition(req, RepositionRequestService.Response())

    def publish_tf(self, cur_tf, timestamp):
        """Publish the ground-truth transform to the TF tree"""
        trans = tesse_ros_bridge.utils.get_translation_part(cur_tf)
        quat = tesse_ros_bridge.utils.get_quaternion(cur_tf)
        self.tf_broadcaster.sendTransform(mk_transform(
                                          trans, quat, timestamp,
                                          self.body_frame_id_gt,
                                          self.world_frame_id))


def main(args=None):
    rclpy.init(args=args)
    node = TesseROSWrapper()
    
    try:
        node.spin_node()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
