from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose


class TrajectoryPlotter(Node):
    def __init__(self):
        super().__init__('trajectory_plotter')

        self.desired = None
        self.actual = []

        self.create_subscription(
            Float64MultiArray,
            'desired_trajectory',
            self.desired_callback,
            10
        )

        self.pose_ee_subscription = self.create_subscription(
            Pose,
            'unity_end_effector_pose',
            self.pose_ee_feedback_callback,
            10
        )

    def pose_ee_feedback_callback(self, msg):
        x = msg.position.x
        y = msg.position.y
        z = msg.position.z

        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        self.actual.append([x, y, z, qx, qy, qz, qw])

    def desired_callback(self, msg):
        data = np.array(msg.data)
        N = len(data) // 7   # XYZ + Qx Qy Qz Qw

        self.desired = data.reshape(N, 7)
        self.get_logger().info(f"Ricevuti {N} punti desiderati")



    def plot(self):
        if self.desired is None or len(self.actual) == 0:
            print("Dati insufficienti per il plot")
            return

        actual = np.array(self.actual)
        # estrai quaternioni
        q_des = self.desired[:, 3:7]
        q_act = actual[:, 3:7]

        # converti in RPY
        rpy_des = R.from_quat(q_des).as_euler('xyz', degrees=True)
        rpy_act = R.from_quat(q_act).as_euler('xyz', degrees=True)

        plt.figure(figsize=(12, 10))

        # Posizione
        plt.subplot(4, 1, 1)
        plt.plot(self.desired[:,0], label='X_des')
        plt.plot(actual[:,0], label='X_act')
        plt.legend()
        plt.title("Posizione")

        # Velocità
        plt.subplot(4, 1, 2)
        v_des = np.gradient(self.desired, axis=0)
        v_act = np.gradient(actual, axis=0)
        plt.plot(v_des[:,0], label='Vx_des')
        plt.plot(v_act[:,0], label='Vx_act')
        plt.legend()
        plt.title("Velocità")

        # Accelerazione
        plt.subplot(4, 1, 3)
        a_des = np.gradient(v_des, axis=0)
        a_act = np.gradient(v_act, axis=0)
        plt.plot(a_des[:,0], label='Ax_des')
        plt.plot(a_act[:,0], label='Ax_act')
        plt.legend()
        plt.title("Accelerazione")

        # Jerk
        plt.subplot(4, 1, 4)
        j_des = np.gradient(a_des, axis=0)
        j_act = np.gradient(a_act, axis=0)
        plt.plot(j_des[:,0], label='Jx_des')
        plt.plot(j_act[:,0], label='Jx_act')
        plt.legend()
        plt.title("Jerk")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)


        plt.figure(figsize=(12, 10))

        plt.subplot(3,1,1)
        plt.plot(rpy_des[:,0], label='roll_des')
        plt.plot(rpy_act[:,0], label='roll_act')
        plt.legend()
        plt.title("Orientazione X - Roll")

        plt.subplot(3,1,2)
        plt.plot(rpy_des[:,1], label='pitch_des')
        plt.plot(rpy_act[:,1], label='pitch_act')
        plt.legend()
        plt.title("Orientazione Y - Pitch")

        plt.subplot(3,1,3)
        plt.plot(rpy_des[:,2], label='yaw_des')
        plt.plot(rpy_act[:,2], label='yaw_act')
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

    # Mantieni aperte le finestre dei plot
    try:
        while True:
            plt.pause(0.1)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
