from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState
from rclpy.node import Node
import numpy as np
import rclpy
import time
from my_robot_utils import kinematics


class PointsGenerator(Node):
    def __init__(self):
        super().__init__('tm5_generatore_punti')
        
        # Publisher dei waypoint (frame: base_link)
        self.publisher_ = self.create_publisher(Float64MultiArray, 'waypoint', 10)
        
        # READY dal controller (Unity in Play)
        self.ready_to_send = False
        self.create_subscription(Bool, 'robot_ready', self.ready_callback, 10)
        
        # URDF per la cinematica
        self.declare_parameter('robot_description', '')
        self.robot_desc = self.get_parameter('robot_description').get_parameter_value().string_value
        self.kinematics = kinematics.KDLKinematics6DOF(self.robot_desc)

        # Stato attuale dei giunti (da Unity → joint_state_bridge)
        self.current_q = None
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

    def joint_cb(self, msg: JointState):
        # Salvo solo le posizioni (assumo ordine corretto)
        self.current_q = list(msg.position)

    def ready_callback(self, msg: Bool):
        if msg.data:
            self.ready_to_send = True

    def send_waypoint(self, points):
        self.get_logger().info('In attesa del segnale READY dal controller (Unity deve essere in Play)...')

        while rclpy.ok() and not self.ready_to_send:
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

        pts = np.asarray(points, dtype=float)
        flat = [float(len(pts))] + pts.flatten().astype(float).tolist()

        msg = Float64MultiArray()
        msg.data = flat
        self.publisher_.publish(msg)
        self.get_logger().info(f'Handshake completato. Inviati {len(pts)} punti.')
        return True

    def generate_zip(self, start=np.array([0.3, 0.0, 0.15]),
                     end=np.array([0.9, 0.5, 0.15]),
                     num_points=100, amplitude=0.02, frequency=3.0):

        t = np.linspace(0, 1, num_points)
        line = start + np.outer(t, (end - start))

        direction = end[:2] - start[:2]
        direction = direction / np.linalg.norm(direction)

        normal = np.array([-direction[1], direction[0]])
        wave = amplitude * np.sin(2 * np.pi * frequency * t)

        line[:, 0] += wave * normal[0]
        line[:, 1] += wave * normal[1]

        return line

    def linear_segment(self, p_start, p_end, num=40):
        t = np.linspace(0, 1, num)
        pts = p_start + np.outer(t, (p_end - p_start))
        return pts

    def generate_pocket_square2(self, width=0.12, height=0.18, z=0.15,
                                n_per_side=25, noise=0.002,
                                offset_x=0.5, offset_y=0.3):

        w = width
        h = height
        pts = []

        for i in range(n_per_side):
            y = i * (h / n_per_side)
            pts.append([0, y, z])

        t = np.linspace(0, np.pi, n_per_side)
        for ti in t:
            x = (w/2) * (1 - np.cos(ti))
            y = h
            pts.append([x, y, z])

        for i in range(n_per_side):
            y = h - i * (h / n_per_side)
            pts.append([w, y, z])

        t = np.linspace(np.pi, 2*np.pi, n_per_side)
        for ti in t:
            x = (w/2) * (1 - np.cos(ti))
            y = 0
            pts.append([x, y, z])

        pts = np.array(pts)
        pts[:, 0] += noise * np.random.randn(len(pts))
        pts[:, 1] += noise * np.random.randn(len(pts))
        pts[:, 0] += offset_x
        pts[:, 1] += offset_y

        return pts

    def build_full_trajectory(self):
        # Aspetto di avere una q reale
        self.get_logger().info("In attesa di /joint_states per ottenere q_start reale...")
        while rclpy.ok() and self.current_q is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

        q_start = np.array(self.current_q)
        self.get_logger().info(f"q_start reale: {q_start}")

        # FK per ottenere X_home in base_link
        T_home = self.kinematics.fk_6dof(q_start)
        X_home = self.kinematics.position_from_T(T_home)  # shape (3,)
        self.get_logger().info(f"X_home: {X_home}")

        zip_traj = self.generate_zip()
        tasca = self.generate_pocket_square2()

        self.get_logger().info(f"X_home = {X_home}, primo waypoint = {zip_traj[0]}")


        approach_traj1 = self.linear_segment(X_home, zip_traj[0])
        retreat_traj1 = self.linear_segment(zip_traj[-1], X_home)

        approach_traj2 = self.linear_segment(X_home, tasca[0])
        retreat_traj2 = self.linear_segment(tasca[-1], X_home)
        full = np.vstack([zip_traj, tasca])


        return full


def main():
    rclpy.init()
    node = PointsGenerator()
    try:
        traj = node.build_full_trajectory()
        node.send_waypoint(traj)
    except KeyboardInterrupt:
        node.get_logger().info("Chiusura forzata dall'utente.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
