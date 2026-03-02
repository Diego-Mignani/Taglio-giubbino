import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

class TrajectoryPlotter(Node):
    def __init__(self):
        super().__init__('trajectory_plotter')

    def plotta_traiettoria(self):
        file_path = os.path.expanduser("~/ros2_ws/log_traj/traj_log.csv")
        data = np.genfromtxt(file_path, delimiter=",", skip_header=1)

        # -------------------------
        # Parsing colonne
        # -------------------------
        t = data[:, 0]

        Xd      = data[:, 1:4]
        Xd_dot  = data[:, 4:7]
        Xd_ddot = data[:, 7:10]

        X       = data[:, 10:13]
        Xdot    = data[:, 13:16]

        Qd = data[:, 16:20]   # qx qy qz qw
        Q  = data[:, 20:24]

        omega = data[:, 24:27] if data.shape[1] >= 27 else None

        # -------------------------
        # Derivate numeriche reali
        # -------------------------
        acc_act  = np.gradient(Xdot[:, 0], t)
        jerk_act = np.gradient(acc_act, t)
        jerk_des = np.gradient(Xd_ddot[:, 0], t)

        # -------------------------
        # ORIENTAZIONI → EULER
        # -------------------------
        eul_des = R.from_quat(Qd).as_euler('xyz', degrees=True)
        eul_act = R.from_quat(Q).as_euler('xyz', degrees=True)

        # unwrap per evitare gradini ±180°
        eul_des_unw = np.unwrap(eul_des * np.pi/180, axis=0) * 180/np.pi
        eul_act_unw = np.unwrap(eul_act * np.pi/180, axis=0) * 180/np.pi

        eul_err = eul_des_unw - eul_act_unw

        # -------------------------
        # Errore orientazione come angolo equivalente
        # -------------------------
        dot = np.sum(Qd * Q, axis=1)
        dot = np.clip(dot, -1.0, 1.0)

        # -------------------------
        # PLOT POSIZIONE / VELOCITÀ / ACC / JERK
        # -------------------------
        plt.figure(figsize=(14, 12))

        plt.subplot(4, 1, 1)
        plt.plot(t, Xd[:, 0], label="X_des")
        plt.plot(t, X[:, 0], label="X_act")
        plt.legend(); plt.title("Posizione X")

        plt.subplot(4, 1, 2)
        plt.plot(t, Xd_dot[:, 0], label="Vx_des")
        plt.plot(t, Xdot[:, 0], label="Vx_act")
        plt.legend(); plt.title("Velocità X")

        plt.subplot(4, 1, 3)
        plt.plot(t, Xd_ddot[:, 0], label="Ax_des")
        plt.plot(t, acc_act, label="Ax_act")
        plt.legend(); plt.title("Accelerazione X")

        plt.subplot(4, 1, 4)
        plt.plot(t, jerk_des, label="Jx_des")
        plt.plot(t, jerk_act, label="Jx_act")
        plt.legend(); plt.title("Jerk X")

        plt.tight_layout()
        plt.show()

        # -------------------------
        # PLOT ORIENTAZIONI (EULER UNWRAPPED)
        # -------------------------
        plt.figure(figsize=(14, 12))
        labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]

        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(t, eul_des_unw[:, i], label=f"{labels[i]} des")
            plt.plot(t, eul_act_unw[:, i], label=f"{labels[i]} act")
            plt.plot(t, eul_err[:, i], '--', label=f"{labels[i]} err")
            plt.legend()
            plt.title(f"Orientazione – {labels[i]} (unwrap)")

        plt.tight_layout()
        plt.show()

        # -------------------------
        # PLOT QUATERNIONI
        # -------------------------
        plt.figure(figsize=(14, 10))
        q_labels = ["qx", "qy", "qz", "qw"]

        for i in range(4):
            plt.subplot(4, 1, i+1)
            plt.plot(t, Qd[:, i], label=f"{q_labels[i]} des")
            plt.plot(t, Q[:, i], label=f"{q_labels[i]} act")
            plt.legend()
            plt.title(f"Quaternione – {q_labels[i]}")

        plt.tight_layout()
        plt.show()



def main():
    rclpy.init()
    node = TrajectoryPlotter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    print("Plotting…")
    node.plotta_traiettoria()

    node.destroy_node()
    rclpy.shutdown()
