from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import os
from my_robot_utils import kinematics
from ament_index_python.packages import get_package_share_directory


class TrajectoryPlotter(Node):
    def __init__(self):
        super().__init__('trajectory_plotter')

        # URDF dal package
        description_package = get_package_share_directory('tm_description')
        urdf_file = os.path.join(description_package, 'urdf', 'tm5-900.urdf')
        with open(urdf_file, 'r') as infp:
            robot_desc = infp.read()
        self.kin = kinematics.KDLKinematics6DOF(robot_desc)

        # buffer dati desiderati
        self.des_pos = None
        self.des_vel = None
        self.des_acc = None
        self.des_quat = None
        self.t_des = None

        # buffer giunti
        self.q_list = []
        self.qdot_list = []

        # dt Unity
        self.dt_act = 0.033

        # subscriber
        self.create_subscription(Float64MultiArray, 'trajectory', self.desired_callback, 10)
        self.create_subscription(JointState, 'unity_joint_feedback', self.joint_callback, 10)

    # -------------------------
    # CALLBACKS
    # -------------------------

    def joint_callback(self, msg: JointState):
        if len(msg.position) == 6:
            self.q_list.append(np.array(msg.position))

        if len(msg.velocity) == 6:
            self.qdot_list.append(np.array(msg.velocity))
        else:
            # se Unity non manda velocity, metti zero
            self.qdot_list.append(np.zeros(6))

    def desired_callback(self, msg):
        data = np.array(msg.data)
        N = len(data) // 14
        reshaped = data.reshape(N, 14)

        self.t_des = reshaped[:, 0]
        self.des_pos = reshaped[:, 1:4]
        self.des_vel = reshaped[:, 4:7]
        self.des_acc = reshaped[:, 7:10]
        self.des_quat = reshaped[:, 10:14]

        self.get_logger().info(f"Ricevuti {N} punti desiderati")

    # -------------------------
    # PLOTTING
    # -------------------------

    def plot(self):
        if self.des_pos is None or len(self.q_list) == 0:
            print("Dati insufficienti per il plot")
            return

        # allinea q e qdot
        N = min(len(self.q_list), len(self.qdot_list))
        q_arr = np.array(self.q_list[:N])
        qdot_arr = np.array(self.qdot_list[:N])

        # tempo attuale grezzo
        t_act = np.arange(N) * self.dt_act
        t_des = self.t_des.copy()

        # allineamento temporale: taglia l'attuale prima dell'inizio del desiderato
        t0_des = t_des[0]
        mask_act = t_act >= t0_des
        t_act = t_act[mask_act] - t0_des
        q_arr = q_arr[mask_act]
        qdot_arr = qdot_arr[mask_act]

        # shift anche il desiderato
        t_des = t_des - t0_des

        # ricostruzione posizione e velocità cartesiana
        X_act = []
        Xdot_act = []

        for q, qdot in zip(q_arr, qdot_arr):
            T = self.kin.fk_6dof(q)
            pos = self.kin.position_from_T(T)

            J = self.kin.get_full_jacobian(q)
            J_pos = J[:3, :]

            vel = J_pos @ qdot

            X_act.append(pos)
            Xdot_act.append(vel)

        X_act = np.array(X_act)
        Xdot_act = np.array(Xdot_act)
        if Xdot_act.ndim == 1:
            Xdot_act = Xdot_act.reshape(1, -1)

        # accelerazione e jerk attuali
        acc_act = np.gradient(Xdot_act[:, 0], self.dt_act)
        jerk_act = np.gradient(acc_act, self.dt_act)

        # jerk desiderato
        jerk_des = np.gradient(self.des_acc[:, 0], t_des)

        # -------------------------
        # POSIZIONE / VELOCITÀ / ACC / JERK
        # -------------------------

        plt.figure(figsize=(12, 10))

        plt.subplot(4, 1, 1)
        plt.plot(t_des, self.des_pos[:, 0], label='X_des')
        plt.plot(t_act, X_act[:, 0], label='X_act')
        plt.legend(); plt.title("Posizione")

        plt.subplot(4, 1, 2)
        plt.plot(t_des, self.des_vel[:, 0], label='Vx_des')
        plt.plot(t_act, Xdot_act[:, 0], label='Vx_act')
        plt.legend(); plt.title("Velocità")

        plt.subplot(4, 1, 3)
        plt.plot(t_des, self.des_acc[:, 0], label='Ax_des')
        plt.plot(t_act, acc_act, label='Ax_act')
        plt.legend(); plt.title("Accelerazione")

        plt.subplot(4, 1, 4)
        plt.plot(t_des, jerk_des, label='Jx_des')
        plt.plot(t_act, jerk_act, label='Jx_act')
        plt.legend(); plt.title("Jerk")

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
    node.plot()

    node.destroy_node()
    rclpy.shutdown()
