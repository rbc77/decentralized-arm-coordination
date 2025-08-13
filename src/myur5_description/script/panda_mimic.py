#!/usr/bin/env python3
import rospy
import numpy as np
import time

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import CameraInfo, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf2_ros
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel

import tf_conversions


class PandaIBVSFollower:
    def __init__(self):
        rospy.init_node("panda_follower_ibvs", anonymous=True)

        # Parameters
        self.ns = rospy.get_namespace().strip("/")
        self.centroid_topic = rospy.get_param("~centroid_topic", "/robot2/green_target/centroid")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/robot2/camera/camera_info")
        self.joint_states_topic = rospy.get_param("~joint_states_topic", "/robot2/joint_states")
        self.traj_cmd_topic = rospy.get_param("~traj_cmd_topic", "/robot2/panda_controller/command")

        self.base_link = rospy.get_param("~base_link", "robot2_tf/panda_link0")
        self.ee_link = rospy.get_param("~ee_link", "robot2_tf/panda_link8")
        self.camera_frame = rospy.get_param("~camera_frame", "robot2_tf/camera_optical_frame")

        self.joint_names = rospy.get_param(
            "~joint_names",
            ["panda_joint1", "panda_joint2", "panda_joint3",
             "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
        )

        # Control gains and limits
        self.lambda_gain = rospy.get_param("~lambda", 0.5)
        self.lin_speed_max = rospy.get_param("~lin_speed_max", 0.08)
        self.ang_speed_max = rospy.get_param("~ang_speed_max", 0.20)
        self.joint_vel_max = rospy.get_param("~joint_vel_max", 0.8)
        self.dt = rospy.get_param("~control_dt", 0.05)
        self.target_timeout = rospy.get_param("~target_timeout", 0.5)

        # Bounding box motion detection parameters
        self.bb_motion_threshold = rospy.get_param("~bb_motion_threshold", 3.0)  # pixels
        self.bb_stillness_time = rospy.get_param("~bb_stillness_time", 0.2)      # seconds
        self.bb_history_size = rospy.get_param("~bb_history_size", 5)            # frames

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.image_w = None
        self.image_h = None

        # State
        self.last_centroid = None  # [u, v, Z]
        self.last_centroid_time = 0.0
        self.q = np.zeros(len(self.joint_names))
        self.q_recv = False

        # Bounding box motion tracking
        self.centroid_history = []
        self.last_motion_time = time.time()

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # KDL initialization
        self.robot = None
        self.kdl_chain = None
        self.kdl_fk = None
        self.kdl_jac_solver = None
        self._init_kdl()

        # Subscribers
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        rospy.Subscriber(self.centroid_topic, Float64MultiArray, self._centroid_cb, queue_size=10)
        rospy.Subscriber(self.joint_states_topic, JointState, self._joint_states_cb, queue_size=10)

        # Publisher
        self.traj_pub = rospy.Publisher(self.traj_cmd_topic, JointTrajectory, queue_size=10)

        rospy.loginfo("Panda IBVS follower started with bounding box motion detection")
        rospy.loginfo("Motion threshold: %.1f px, Stillness time: %.2f s", 
                      self.bb_motion_threshold, self.bb_stillness_time)

    def _init_kdl(self):
        urdf_param = rospy.get_param("~urdf_param", "/robot2/robot_description")
        urdf_xml = rospy.get_param(urdf_param, None)
        if urdf_xml is None:
            rospy.logwarn("URDF not found at %s; trying /robot_description", urdf_param)
            urdf_xml = rospy.get_param("/robot_description", None)
        if urdf_xml is None:
            rospy.logerr("URDF not found on parameter server; KDL will not initialize.")
            return
        try:
            self.robot = URDF.from_xml_string(urdf_xml)
            ok, tree = treeFromUrdfModel(self.robot)
            if not ok:
                rospy.logerr("Failed to parse URDF into KDL tree.")
                return
            base_no_prefix = self.base_link.split("/")[-1]
            ee_no_prefix = self.ee_link.split("/")[-1]
            self.kdl_chain = tree.getChain(base_no_prefix, ee_no_prefix)
            self.kdl_fk = kdl.ChainFkSolverPos_recursive(self.kdl_chain)
            self.kdl_jac_solver = kdl.ChainJntToJacSolver(self.kdl_chain)
            rospy.loginfo("KDL chain initialized: %s -> %s", base_no_prefix, ee_no_prefix)
        except Exception as e:
            rospy.logerr("KDL init failed: %s", str(e))

    def _camera_info_cb(self, msg: CameraInfo):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.image_w = msg.width
        self.image_h = msg.height

    def _centroid_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            return
        u = float(msg.data[0])
        v = float(msg.data[1])
        Z = float(msg.data[2]) if len(msg.data) >= 3 else 0.6
        self.last_centroid = np.array([u, v, Z], dtype=float)
        self.last_centroid_time = time.time()

        # Update centroid history for motion detection
        self.centroid_history.append((u, v, self.last_centroid_time))
        if len(self.centroid_history) > self.bb_history_size:
            self.centroid_history.pop(0)

    def _joint_states_cb(self, msg: JointState):
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        q_list = []
        for n in self.joint_names:
            if n not in name_to_pos:
                return
            q_list.append(name_to_pos[n])
        self.q = np.array(q_list, dtype=float)
        self.q_recv = True

    def _is_bounding_box_moving(self):
        """Check if bounding box (centroid) is moving based on recent history"""
        if len(self.centroid_history) < 2:
            return True  # Assume moving if insufficient data
        
        now = time.time()
        recent_centroids = [(u, v) for u, v, t in self.centroid_history 
                           if now - t <= self.bb_stillness_time]
        
        if len(recent_centroids) < 2:
            return True  # Assume moving if insufficient recent data
        
        # Calculate maximum displacement in recent history
        max_displacement = 0.0
        for i in range(len(recent_centroids)):
            for j in range(i + 1, len(recent_centroids)):
                u1, v1 = recent_centroids[i]
                u2, v2 = recent_centroids[j]
                displacement = np.sqrt((u2 - u1)**2 + (v2 - v1)**2)
                max_displacement = max(max_displacement, displacement)
        
        is_moving = max_displacement > self.bb_motion_threshold
        
        if is_moving:
            self.last_motion_time = now
        
        # Log motion state occasionally
        rospy.loginfo_throttle(2.0, "BB motion: %.1f px, moving: %s", 
                              max_displacement, "YES" if is_moving else "NO")
        
        return is_moving

    def _build_interaction_matrix_point(self, u, v, Z):
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return None
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        Z = max(0.05, float(Z))
        L = np.array([
            [-1.0 / Z, 0.0, x / Z, x * y, -(1.0 + x * x), y],
            [0.0, -1.0 / Z, y / Z, 1.0 + y * y, -x * y, -x]
        ], dtype=float)
        return L

    def _compute_camera_twist(self, u, v, Z):
        # Check if bounding box is moving
        if not self._is_bounding_box_moving():
            rospy.loginfo_throttle(1.0, "Bounding box is still - holding robot position")
            return np.zeros(6, dtype=float)

        # Compute standard IBVS control if bounding box is moving
        ex = (u - self.cx)
        ey = (v - self.cy)
        e = np.array([ex, ey], dtype=float)

        L = self._build_interaction_matrix_point(u, v, Z)
        if L is None:
            return None

        Lt = L.T
        mu = 1e-6
        Lp = Lt.dot(np.linalg.inv(L.dot(Lt) + mu * np.eye(2)))
        v_cam = -self.lambda_gain * Lp.dot(e)

        v_lin = v_cam[0:3]
        v_ang = v_cam[3:6]
        nlin = np.linalg.norm(v_lin)
        nang = np.linalg.norm(v_ang)
        if nlin > self.lin_speed_max:
            v_lin *= (self.lin_speed_max / max(1e-9, nlin))
        if nang > self.ang_speed_max:
            v_ang *= (self.ang_speed_max / max(1e-9, nang))
        return np.hstack([v_lin, v_ang])

    def _kdl_jacobian(self, q_np: np.ndarray):
        if self.kdl_jac_solver is None:
            return None
        nj = self.kdl_chain.getNrOfJoints()
        q_kdl = kdl.JntArray(nj)
        for i in range(nj):
            q_kdl[i] = float(q_np[i])
        J = kdl.Jacobian(nj)
        self.kdl_jac_solver.JntToJac(q_kdl, J)
        J_np = np.zeros((6, nj), dtype=float)
        for r in range(6):
            for c in range(nj):
                J_np[r, c] = J[r, c]
        return J_np

    def _twist_camera_to_base(self, v_cam):
        if v_cam is None:
            return None
        try:
            t = self.tf_buffer.lookup_transform(self.base_link, self.camera_frame, rospy.Time(0), rospy.Duration(0.3))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "TF lookup base<-camera failed: %s", str(e))
            return None
        q = t.transform.rotation
        R = tf_conversions.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[0:3, 0:3]
        v_lin_b = R.dot(v_cam[0:3])
        v_ang_b = R.dot(v_cam[3:6])
        return np.hstack([v_lin_b, v_ang_b])

    def _compute_qdot(self, v_base, q_curr):
        J = self._kdl_jacobian(q_curr)
        if J is None:
            return None
        Jt = J.T
        mu = 1e-6
        JJt = J.dot(Jt)
        v = v_base.reshape((6, 1))
        qdot = Jt.dot(np.linalg.inv(JJt + mu * np.eye(6))).dot(v)
        qdot = np.clip(qdot.flatten(), -self.joint_vel_max, self.joint_vel_max)
        return qdot

    def _publish_trajectory_point(self, q_next):
        jt = JointTrajectory()
        jt.joint_names = list(self.joint_names)
        p = JointTrajectoryPoint()
        p.positions = [float(qi) for qi in q_next.tolist()]
        p.velocities = [0.0] * len(self.joint_names)
        p.time_from_start = rospy.Duration.from_sec(self.dt * 1.1)
        jt.points = [p]
        self.traj_pub.publish(jt)

    def spin(self):
        rate = rospy.Rate(1.0 / self.dt)
        rospy.sleep(1.0)

        while not rospy.is_shutdown():
            now = time.time()
            ok_measure = self.last_centroid is not None and (now - self.last_centroid_time) <= self.target_timeout
            ok_cam = (self.fx is not None and self.cx is not None and self.cy is not None)
            ok_kdl = (self.kdl_jac_solver is not None)

            if not (ok_measure and ok_cam and self.q_recv and ok_kdl):
                if self.q_recv:
                    self._publish_trajectory_point(self.q)
                rate.sleep()
                continue

            u, v, Z = self.last_centroid
            v_cam = self._compute_camera_twist(u, v, Z)
            if v_cam is None:
                if self.q_recv:
                    self._publish_trajectory_point(self.q)
                rate.sleep()
                continue

            v_base = self._twist_camera_to_base(v_cam)
            if v_base is None:
                if self.q_recv:
                    self._publish_trajectory_point(self.q)
                rate.sleep()
                continue

            q_curr = self.q.copy()

            if np.allclose(v_base, 0.0, atol=1e-6):
                self._publish_trajectory_point(q_curr)
                rate.sleep()
                continue

            qdot = self._compute_qdot(v_base, q_curr)
            if qdot is None:
                self._publish_trajectory_point(q_curr)
                rate.sleep()
                continue

            q_next = q_curr + qdot * self.dt
            self._publish_trajectory_point(q_next)
            rate.sleep()


if __name__ == "__main__":
    node = PandaIBVSFollower()
    node.spin()
