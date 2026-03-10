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
        self.get_logger().info("### VERSIONE TRAJECTORY MANAGER 2026-03-09 ###")

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

        self.build_and_publish(pts)


    # ----------------------------------------------------------
    # MOVEIT2 PLANNING via servizio cartesiano
    # ----------------------------------------------------------
    def plan_cartesian(self, points):
        self.get_logger().info("DEBUG: sono dentro plan_cartesian")
        self.get_logger().info(f"Chiamo MoveIt con {len(points)} waypoint")

        req = GetCartesianPath.Request()
        req.group_name = self.group_name
        req.link_name = self.eef_link
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False




        # Usa lo stato reale di Unity come start_state

        # CONFIG DI TEST, FISSA
        q_start = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]  # rad

        rs = RobotState()
        rs.joint_state.name = self.joint_names
        rs.joint_state.position = q_start
        req.start_state = rs
        self.get_logger().info(f"Start_state di TEST: {q_start}")


        for p in points:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])

            # TEST: niente orientazione “furba”
            # 1) identità (end-effector allineato all’asse z del link)
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0

            req.waypoints.append(pose)


        future = self.cartesian_client.call_async(req)

        try:
            self.get_logger().info("[TM] prima di spin_until_future_complete")
            rclpy.spin_until_future_complete(self, future)
            self.get_logger().info("[TM] dopo spin_until_future_complete")

        except Exception as e:
            self.get_logger().error(f"[TM] Eccezione in spin_until_future_complete: {e}")
            return None

        if not future.result():
            self.get_logger().error("Chiamata a /compute_cartesian_path fallita (future.result() è None)")
            return None

        res = future.result()
        self.get_logger().info(
            f"MoveIt ha pianificato: fraction={res.fraction:.3f}, "
            f"num_points={len(res.solution.joint_trajectory.points)}, "
            f"error_code={res.error_code.val}"
        )

        if res.fraction < 0.99:
            self.get_logger().warn(
                f"Solo {res.fraction*100:.1f}% della traiettoria cartesiana pianificata"
            )

        traj = res.solution.joint_trajectory
        if not traj.points:
            self.get_logger().error("Traiettoria vuota da MoveIt2")
            return None

        return traj

    # ----------------------------------------------------------
    # COSTRUZIONE TRAIETTORIA COMPLETA
    # ----------------------------------------------------------
        
    def build_and_publish(self, pts):
        self.get_logger().info("[TM] build_and_publish: INIZIO")

        # 1) Pianificazione MoveIt
        traj = self.plan_cartesian(pts)

        if traj is None:
            self.get_logger().error("[TM] Errore nella pianificazione MoveIt2 (traj è None)")
            return

        if not hasattr(traj, "points") or traj.points is None:
            self.get_logger().error("[TM] ERRORE: traj.points è None!")
            return

        self.get_logger().info(f"[TM] build_and_publish: traj.points={len(traj.points)}")

        # 2) Pubblica verso il controller
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
