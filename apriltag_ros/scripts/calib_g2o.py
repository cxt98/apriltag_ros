import rospy
import ros_numpy

from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
from tf import TransformBroadcaster
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool
import argparse
import yaml

import os
import g2o
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R


class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().set_verbose(True)
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


class PoseCalibrator:
    def __init__(self, args) -> None:

        rospy.init_node(args.node_name)

        self.args = args

        self.cam_tag = None
        self.cam2_tag = None
        self.cam2robot_pose = None
        self.cam2cam_pose = None
        
        self.load_tag_yaml(args.tag_yaml_path)
        self._lock = threading.RLock()
        self.pgo = PoseGraphOptimization()

        # subscribe to apriltag pose result
        self.cam_tag_sub = rospy.Subscriber(args.cam_tag_channel, AprilTagDetectionArray, self.subscribe_cam_tag, callback_args='cam', queue_size=1)
        self.cam2_tag_sub = rospy.Subscriber(args.cam2_tag_channel, AprilTagDetectionArray, self.subscribe_cam_tag, callback_args='cam2', queue_size=1)

        # wait for user activation to do calibration
        self.run_cam2robot_calib_sub = rospy.Subscriber(args.cam2robot_activation_channel, Bool, self.cam2robot_calib_callback, queue_size=1)
        self.run_cam2cam_calib_sub = rospy.Subscriber(args.cam2cam_activation_channel, Bool, self.cam2cam_calib_callback, queue_size=1)

        # publish calibrated transform
        self.cam2robot_pub = rospy.Publisher(args.cam2robot_publish_channel, PoseWithCovarianceStamped, queue_size=1)
        self.cam2cam_pub = rospy.Publisher(args.cam2cam_publish_channel, PoseWithCovarianceStamped, queue_size=1)
    
    def load_tag_yaml(self, path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            tags = config['standalone_tags']
            self.tag_id_robot = None
            self.tag_id_others = []
            for t in tags:
                if t['name'] == 'Digit':
                    self.tag_id_robot = t['id']
                else:
                    self.tag_id_others.append(t['id'])
        rospy.loginfo('Tag Info Loaded, robot id: %d; others: %s', self.tag_id_robot, self.tag_id_others)

    def subscribe_cam_tag(self, msg, args):
        with self._lock:
            if args == 'cam':
                self.cam_tag = msg # AprilTagDetectionArray
            else:
                self.cam2_tag = msg

    def publish_calib(self):
        if self.cam2robot_pose is not None:
            self.cam2robot_pose.header.frame_id = "base"
            self.cam2robot_pub.publish(self.cam2robot_pose)
            # rospy.loginfo('Publishing cam2robot Pose')
        if self.cam2cam_pose is not None:
            self.cam2cam_pub.publish(self.cam2cam_pose)

    def cam2robot_calib_callback(self, msg):
        with self._lock:
            if self.cam_tag is not None:
                rospy.loginfo('running cam2robot calibration')
                for t in self.cam_tag.detections:
                    if t.id[0] == self.tag_id_robot:
                        rospy.loginfo('Assigned AprilTag %d to cam2robot Pose and started to publish', self.tag_id_robot)
                        self.cam2robot_pose = t.pose

    def cam2cam_calib_callback(self, msg): # currently work for 2 cameras only
        pass

    def mat44_posestamped(self, mat44):
        p = PoseWithCovarianceStamped()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = 'cam2robot'
        p.pose.position.x = mat44[0, 3]
        p.pose.position.y = mat44[1, 3]
        p.pose.position.z = mat44[2, 3]
        quat = R.from_matrix(mat44[:3, :3]).as_quat()
        p.pose.orientation.x = quat[0]
        p.pose.orientation.y = quat[1]
        p.pose.orientation.z = quat[2]
        p.pose.orientation.w = quat[3]
        return p
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--node_name', type=str, default='Camera_Robot_Pose_Calibrator')
    parser.add_argument('--tag_yaml_path', type=str, default='src/apriltag_ros/apriltag_ros/config/tags.yaml')
    parser.add_argument('--cam2robot_activation_channel', type=str, default='/user/run_cam2robot_calib')
    parser.add_argument('--cam2cam_activation_channel', type=str, default='/user/run_cam2cam_calib')
    parser.add_argument('--cam_tag_channel', type=str, default='/tag_detections', help=' cam2robot calib and 1st camera in cam2cam calib') 
    parser.add_argument('--cam2_tag_channel', type=str, default='/tag_detections', help='2nd camera in cam2cam calib')
    parser.add_argument('--save_calib_path', type=str, default='src/apriltag_ros/apriltag_ros/config/calib.yaml')
    parser.add_argument('--cam2robot_publish_channel', type=str, default='/tf_cam2robot', help='pose of tag on robot in camera\'s frame') 
    parser.add_argument('--cam2cam_publish_channel', type=str, default='/tf_cam2cam', help='pose of 2nd camera in 1st camera\'s frame')
    
    args = parser.parse_args()

    p = PoseCalibrator(args)
    while not rospy.is_shutdown():
        p.publish_calib()
        rospy.sleep(0.001)