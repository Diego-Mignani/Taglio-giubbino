import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from trajectory_msgs.msg import JointTrajectory
import scipy.spatial.transform as tf
from my_robot_utils import kinematics


class TrajectoryManager(Node):
    def __init__(self):
        super().__init__("tm5_gestione_traiettoria")

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

    # ----------------------------------------------------------
    # CALLBACK
    # ----------------------------------------------------------
    def receive_waypoint(self, msg):
        if self.trajectory_ready:
            return

        data = np.array(msg.data)
        N1, N2 = int(data[0]), int(data[1])
        coords = data[2:]

        pts1 = coords[: 3 * N1].reshape(N1, 3)
        pts2 = coords[3 * N1 : 3 * (N1 + N2)].reshape(N2, 3)

        self.build_and_publish(pts1, pts2)

    # ----------------------------------------------------------
    # MOVEIT2 PLANNING via servizio cartesiano
    # ----------------------------------------------------------
    def plan_cartesian(self, points):
        req = GetCartesianPath.Request()
        req.group_name = self.group_name
        req.link_name = self.eef_link
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False  # se vuoi

        # Stato iniziale vuoto → usa lo stato corrente di MoveIt
        req.start_state.is_diff = True

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

        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if not future.result():
            self.get_logger().error("Chiamata a /compute_cartesian_path fallita")
            return None

        res = future.result()

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
    def build_and_publish(self, pts1, pts2):
        traj1 = self.plan_cartesian(pts1)
        traj2 = self.plan_cartesian(pts2)

        if traj1 is None or traj2 is None:
            self.get_logger().error("Errore nella pianificazione MoveIt2")
            return

        # riallinea timestamp di traj2 dopo traj1
        offset = traj1.points[-1].time_from_start
        for p in traj2.points:
            p.time_from_start.sec += offset.sec
            p.time_from_start.nanosec += offset.nanosec

        full = JointTrajectory()
        full.joint_names = traj1.joint_names
        full.points = traj1.points + traj2.points

        self.publish_joint_trajectory(full)
        self.publish_cartesian_trajectory(full)

        self.trajectory_ready = True
        self.get_logger().info(f"Traiettoria inviata ({len(full.points)} punti).")

    # ----------------------------------------------------------
    # PUBBLICAZIONE VERSO IL CONTROLLER (come prima)
    # ----------------------------------------------------------
    def publish_joint_trajectory(self, traj):
        flat = []
        for p in traj.points:
            t = p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
            flat.append(t)
            flat.extend(p.positions)

        msg = Float64MultiArray()
        msg.data = flat
        self.pub_to_controller.publish(msg)

    # ----------------------------------------------------------
    # PUBBLICAZIONE CARTESIANA PER UNITY (come prima)
    # ----------------------------------------------------------
    def publish_cartesian_trajectory(self, traj):
        flat = []
        for p in traj.points:
            q = np.array(p.positions)
            T = self.kin.fk_6dof(q)
            X = self.kin.position_from_T(T)
            R = self.kin.rotation_from_T(T)
            Q = self.kin.quaternion_from_R(R)
            flat.extend([X[0], X[1], X[2], *Q])

        msg = Float64MultiArray()
        msg.data = flat
        self.pub_to_unity.publish(msg)


def main():
    rclpy.init()
    node = TrajectoryManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
