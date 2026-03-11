import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from trajectory_msgs.msg import JointTrajectory
import scipy.spatial.transform as tf
from my_robot_utils import kinematics
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class TrajectoryManager(Node):
    def __init__(self):
        super().__init__("tm5_gestione_traiettoria")
        self.get_logger().info("TrajectoryManager avviato e in ascolto su /waypoint")

        # URDF per la cinematica diretta (come prima)
        self.declare_parameter("robot_description", "")
        self.robot_desc = self.get_parameter("robot_description").value
        self.kin = kinematics.KDLKinematics6DOF(self.robot_desc)

        # Orientazione "punta in giù"
        self.Q_down = tf.Rotation.from_euler("x", 180, degrees=True).as_quat()

        # Sottoscrizione waypoint (come prima)
        self.sub_wp = self.create_subscription(
            Float64MultiArray, "waypoint", self.receive_waypoint, 10
        )

        # Pubblicazioni (come prima)
        self.pub_to_controller = self.create_publisher(
            Float64MultiArray, "trajectory", 10
        )
        self.pub_to_unity = self.create_publisher(
            Float64MultiArray, "desired_trajectory", 10
        )

        # Client per MoveIt2: servizio cartesiano
        self.cartesian_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path"
        )
        self.get_logger().info("In attesa del servizio /compute_cartesian_path...")
        self.cartesian_client.wait_for_service()
        self.get_logger().info("Servizio /compute_cartesian_path disponibile.")

        self.group_name = "tm5_arm"
        self.eef_link = "flange"  # controlla che sia il nome giusto nel tuo URDF/SRDF

        self.trajectory_ready = False
        self.last_q = None
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]  # metti i nomi giusti

        self.sub_js = self.create_subscription(
            JointState, "unity_joint_feedback", self.js_callback, 10
        )

    def js_callback(self, msg):
        self.last_q = list(msg.position)

    # ----------------------------------------------------------
    # CALLBACK
    # ----------------------------------------------------------
    def receive_waypoint(self, msg):
        print("CALLBACK CHIAMATO")
        self.get_logger().info("CALLBACK CHIAMATO")
        self.get_logger().info(f"receive_waypoint chiamata, len={len(msg.data)}")
        if self.trajectory_ready:
            return

        data = np.array(msg.data)
        N = int(data[0])
        coords = data[1:]
        pts = coords.reshape(N, 3)

        # invece di build_and_publish diretto → pianificazione asincrona
        self.plan_cartesian_async(pts)

    # ----------------------------------------------------------
    # MOVEIT2 PLANNING via servizio cartesiano
    # ----------------------------------------------------------
    def plan_cartesian_async(self, points):
        self.get_logger().info("DEBUG: sono dentro plan_cartesian_async")
        self.get_logger().info(f"Chiamo MoveIt con {len(points)} waypoint")

        req = GetCartesianPath.Request()
        req.group_name = self.group_name
        req.link_name = self.eef_link
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False

        # start_state di test
        q_start = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        rs = RobotState()
        rs.joint_state.name = self.joint_names
        rs.joint_state.position = q_start
        req.start_state = rs
        self.get_logger().info(f"Start_state di TEST: {q_start}")

        flat_xyz = []
        for p in points:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])
            pose.orientation.x = self.Q_down[0]
            pose.orientation.y = self.Q_down[1]
            pose.orientation.z = self.Q_down[2]
            pose.orientation.w = self.Q_down[3]
            req.waypoints.append(pose)

            flat_xyz.extend([pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        msg_xyz = Float64MultiArray()
        msg_xyz.data = flat_xyz
        self.pub_to_unity.publish(msg_xyz)

        future = self.cartesian_client.call_async(req)
        future.add_done_callback(self.cartesian_done_cb)
        self.get_logger().info("[TM] Richiesta cartesiana inviata in async")

        
    def cartesian_done_cb(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"[TM] Errore nella risposta di /compute_cartesian_path: {e}")
            return

        if res is None:
            self.get_logger().error("[TM] future.result() è None")
            return

        self.get_logger().info(
            f"[TM] MoveIt ha pianificato: fraction={res.fraction:.3f}, "
            f"num_points={len(res.solution.joint_trajectory.points)}, "
            f"error_code={res.error_code.val}"
        )

        traj = res.solution.joint_trajectory
        if not traj.points:
            self.get_logger().error("[TM] Traiettoria vuota da MoveIt2")
            return

        # qui possiamo riusare build_and_publish come “fase 2”
        self.build_and_publish_from_traj(traj)


    # ----------------------------------------------------------
    # COSTRUZIONE TRAIETTORIA COMPLETA
    # ----------------------------------------------------------
    def build_and_publish_from_traj(self, traj):
        self.get_logger().info("[TM] build_and_publish_from_traj: INIZIO")

        if not hasattr(traj, "points") or traj.points is None:
            self.get_logger().error("[TM] ERRORE: traj.points è None!")
            return

        self.get_logger().info(f"[TM] build_and_publish_from_traj: traj.points={len(traj.points)}")

        self.publish_joint_trajectory(traj)
        self.trajectory_ready = True
        self.get_logger().info(f"[TM] Traiettoria inviata ({len(traj.points)} punti).")

# ----------------------------------------------------------
    # PUBBLICAZIONE VERSO IL CONTROLLER (come prima)
    # ----------------------------------------------------------
    def publish_joint_trajectory(self, traj):
        try:
            flat = []
            dt = 0.1  # 20 Hz
            t = 0.0

            for p in traj.points:
                flat.append(t)
                flat.extend(p.positions)
                t += dt

            msg = Float64MultiArray()
            msg.data = flat
            self.pub_to_controller.publish(msg)

            self.get_logger().info(
                f"[TM] Pubblico traiettoria joint: {len(traj.points)} punti, "
                f"len(flat)={len(flat)}"
            )
        except Exception as e:
            self.get_logger().error(f"[TM] Errore in publish_joint_trajectory: {e}")



def main():
    rclpy.init()
    node = TrajectoryManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
