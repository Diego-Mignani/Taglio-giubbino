from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose


class TrajectoryPlotter(Node):
    def __init__(self):
        super().__init__('trajectory_plotter')

        # buffer dati
        self.desired = None
        self.actual = []

        # tempi di campionamento (IMPOSTALI TU)
        self.dt_des = 0.1   # dt generatore traiettoria
        self.dt_act = 0.1     # dt Unity / controller

        # subscriber
        self.create_subscription(
            Float64MultiArray,
            'trajectory',
            self.desired_callback,
            10
        )

        self.create_subscription(
            Pose,
            'unity_end_effector_pose',
            self.pose_ee_feedback_callback,
            10
        )

    # -------------------------
    # CALLBACKS
    # -------------------------

    def pose_ee_feedback_callback(self, msg):
        self.actual.append([
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

    def desired_callback(self, msg):
        data = np.array(msg.data)
        N = len(data) // 14
        reshaped = data.reshape(N, 14)

        self.t_des = reshaped[:, 0]
        self.desired = reshaped[:, 1:4]
        self.des_vel = reshaped[:, 4:7]
        self.des_acc = reshaped[:, 7:10]
        self.des_quat = reshaped[:, 10:14]


        self.get_logger().info(f"Ricevuti {N} punti desiderati")


    # -------------------------
    # PLOTTING
    # -------------------------

    def plot(self):
        if self.desired is None or len(self.actual) == 0:
            print("Dati insufficienti per il plot")
            return

        actual = np.array(self.actual)
        t_des = self.t_des

        # asse temporale corretto
        t_act = np.arange(len(actual)) * self.dt_act

        # --- rimuovi timestamp duplicati PRIMA di tutto ---
        mask = np.diff(t_des) > 1e-6
        mask = np.insert(mask, 0, True)  # mantieni il primo punto

        t_des = t_des[mask]
        self.desired = self.desired[mask]
        self.des_vel = self.des_vel[mask]
        self.des_acc = self.des_acc[mask]
        self.des_quat = self.des_quat[mask]

        # estrai quaternioni (già filtrati)
        q_des = self.des_quat
        q_act = actual[:, 3:7]

        # converti in RPY
        rpy_des = R.from_quat(q_des).as_euler('xyz', degrees=True)
        rpy_act = R.from_quat(q_act).as_euler('xyz', degrees=True)

        # velocità / accelerazione / jerk desiderati
        v_des = self.des_vel[:, 0]
        a_des = self.des_acc[:, 0]

        j_des = np.zeros_like(a_des)
        for i in range(1, len(a_des)):
            dt = t_des[i] - t_des[i-1]
            if dt > 1e-6:
                j_des[i] = (a_des[i] - a_des[i-1]) / dt
            else:
                j_des[i] = j_des[i-1]

        # -------------------------
        # POSIZIONE / VELOCITÀ / ACC / JERK
        # -------------------------

        plt.figure(figsize=(12, 10))

        # Posizione
        plt.subplot(4, 1, 1)
        plt.plot(t_des, self.desired[:, 0], label='X_des')
        plt.plot(t_act, actual[:, 0], label='X_act')
        plt.legend()
        plt.title("Posizione")

        # Velocità
        plt.subplot(4, 1, 2)
        v_act = np.gradient(actual[:, 0], self.dt_act)
        plt.plot(t_des, v_des, label='Vx_des')
        plt.plot(t_act, v_act, label='Vx_act')
        plt.legend()
        plt.title("Velocità")

        # Accelerazione
        plt.subplot(4, 1, 3)
        a_act = np.gradient(v_act, self.dt_act)
        plt.plot(t_des, a_des, label='Ax_des')
        plt.plot(t_act, a_act, label='Ax_act')
        plt.legend()
        plt.title("Accelerazione")

        # Jerk
        plt.subplot(4, 1, 4)
        j_act = np.gradient(a_act, self.dt_act)
        plt.plot(t_des, j_des, label='Jx_des')
        plt.plot(t_act, j_act, label='Jx_act')
        plt.legend()
        plt.title("Jerk")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        # -------------------------
        # ORIENTAZIONE
        # -------------------------

        plt.figure(figsize=(12, 10))

        plt.subplot(3, 1, 1)
        plt.plot(t_des, rpy_des[:, 0], label='roll_des')
        plt.plot(t_act, rpy_act[:, 0], label='roll_act')
        plt.legend()
        plt.title("Orientazione X - Roll")

        plt.subplot(3, 1, 2)
        plt.plot(t_des, rpy_des[:, 1], label='pitch_des')
        plt.plot(t_act, rpy_act[:, 1], label='pitch_act')
        plt.legend()
        plt.title("Orientazione Y - Pitch")

        plt.subplot(3, 1, 3)
        plt.plot(t_des, rpy_des[:, 2], label='yaw_des')
        plt.plot(t_act, rpy_act[:, 2], label='yaw_act')
        plt.legend()
        plt.title("Orientazione Z - Yaw")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)


def main():
    rclpy.init()
    node = TrajectoryPlotter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    print("Plotting…")
    node.plot()

    # mantieni aperte le finestre
    try:
        while True:
            plt.pause(0.1)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
