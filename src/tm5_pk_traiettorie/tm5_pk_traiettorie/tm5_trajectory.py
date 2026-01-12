from scipy.interpolate import splev, splrep
import rclpy
from rclpy.node import Node
import numpy as np
import numpy as np
from scipy.interpolate import splrep, splev
from .types import CartesianTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from my_robot_utils import kinematics
import scipy.spatial.transform as tf

class TrajectoryManager(Node):
    def __init__(self):
        super().__init__('tm5_gestione_traiettoria')
        self.dt = 0.033
        self.received_waypoint = None
        self.trajectory_ready = False

        # 1. PARAMETRI E KDL
        self.declare_parameter('robot_description', '')
        self.robot_desc = self.get_parameter('robot_description').get_parameter_value().string_value
        

        self.kinematics = kinematics.KDLKinematics6DOF(self.robot_desc)

        self.q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Da cambiare
        self.Q_down = tf.Rotation.from_euler('x', -90, degrees=True).as_quat()
        
        self.T_home = self.kinematics.fk_6dof(self.q_home)  
        self.X_home = self.kinematics.position_from_T(self.T_home)
        self.R_home = self.kinematics.rotation_from_T(self.T_home)
        self.Q_home = self.kinematics.quaternion_from_R(self.R_home)
        # Subscriber per i waypoint generati dal PointsGenerator
        self.sub_trajectory = self.create_subscription(
            Float64MultiArray, 
            'waypoint', 
            self.receive_waypoint, 
            10)
        # Publisher verso il RobotController
        self.pub_to_controller = self.create_publisher(
            Float64MultiArray, 'trajectory', 10)
        # Publisher per visualizzare in Unity la traiettoria desiderata
        self.visualizza_unity = self.create_publisher(Float64MultiArray, "desired_trajectory", 10)

    def receive_waypoint(self, msg):
        """
        Callback per ricevere i punti della traiettoria

        :param msg: Float64MultiArray contenente i punti della traiettoria
        """
        # Ignora se la traiettoria è già stata processata e per errore il publisher invia di nuovo
        if self.trajectory_ready:
            return
        # Ricostruiamo la matrice N x 3
        #data = np.array(msg.data)
        #self.received_waypoint = data.reshape(-1, 3) 
        #self.get_logger().info('Waypoint ricevuti.')
        #self.trajectory_builder()


        data = np.array(msg.data)

        N1 = int(data[0])
        N2 = int(data[1])

        coords = data[2:]  # tutto il resto

        self.pts1 = coords[:3*N1].reshape(N1, 3)
        self.pts2 = coords[3*N1:3*(N1+N2)].reshape(N2, 3)
        self.trajectory_builder()

        
    def trajectory_builder(self):
        """
        Costruisce la traiettoria completa con approach e retreat.
        
        """

        #self.tracking_traj = self.plan_from_points(self.received_waypoint)

        self.tracking_traj1 = self.plan_from_points(self.pts1)
        self.tracking_traj2 = self.plan_from_points(self.pts2)

        X_first1 = self.tracking_traj1[0].X
        X_last1 = self.tracking_traj1[-1].X
        X_first2 = self.tracking_traj2[0].X
        X_last2 = self.tracking_traj2[-1].X

        # -------------------------------
        # APPROACH (home → primo punto)
        # -------------------------------
        approach_traj1 = self.linear_cartesian_segment(
            self.X_home, X_first1, self.dt, duration=1.5
        )

        # -------------------------------
        # RETREAT (ultimo punto → home)
        # -------------------------------
        retreat_traj1 = self.linear_cartesian_segment(
            X_last1, self.X_home, self.dt, duration=1.5
        )

        # -------------------------------
        # APPROACH (home → primo punto)
        # -------------------------------
        approach_traj2 = self.linear_cartesian_segment(
            self.X_home, X_first2, self.dt, duration=1.5
        )

        # -------------------------------
        # RETREAT (ultimo punto → home)
        # -------------------------------
        retreat_traj2 = self.linear_cartesian_segment(
            X_last2, self.X_home, self.dt, duration=1.5
        )



        # traiettoria totale: approach + tracking + retreat
        full_traj = approach_traj1 + self.tracking_traj1 + retreat_traj1 + approach_traj2 + self.tracking_traj2 + retreat_traj2
        # --- SERIALIZZAZIONE PER ROS 2 ---
        # Creiamo un'unica lista piatta: [t1, x1, y1, z1, vx1, vy1, vz1, ax1, ay1, az1, t2, x2, ...]
        flat_data = []
        for p in full_traj:
            flat_data.extend([p.t, *p.X, *p.Xdot, *p.Xddot])
        
        msg = Float64MultiArray()
        msg.data = flat_data
        self.pub_to_controller.publish(msg)
        self.get_logger().info(f'Inviata traiettoria di {len(full_traj)} punti.')
        self.trajectory_ready = True
        # Pubblica solo X,Y,Z per Unity
        flat_xyz = []
        for p in full_traj:
            flat_xyz.extend([p.X[0], p.X[1], p.X[2]])

        msg_xyz = Float64MultiArray()
        msg_xyz.data = flat_xyz
        self.visualizza_unity.publish(msg_xyz)

 
 
    def plan_from_points(self, points_3d, vmax=0.2, amax=0.5, smooth=0.01):
        """
        Genera una traiettoria liscia e time-scaled dai punti dati.

        :param points_3d: array Nx3 di punti grezzi (dalla rete neurale o mock)
        :param vmax: velocità massima lungo la traiettoria (m/s)
        :param amax: accelerazione massima lungo la traiettoria (m/s^2)
        :param smooth: parametro di smoothing della spline
        """
        # 1) ascissa curvilinea s_i
        s = self._compute_s(points_3d)
        L = s[-1]

        # 2) spline smoothing
        tck_x = splrep(s, points_3d[:,0], s=smooth)
        tck_y = splrep(s, points_3d[:,1], s=smooth)
        tck_z = splrep(s, points_3d[:,2], s=smooth)

        # 3) time-scaling trapezoidale
        t, s_t, v_t, a_t = self._trapezoidal_time_scaling(L, vmax, amax)

        # 4) valutazione Xd, Xd_dot, Xd_ddot
        Xd, Xd_dot, Xd_ddot = self.evaluate_trajectory(tck_x, tck_y, tck_z,
                                                        s_t, v_t, a_t)

        # 5) costruzione lista di punti
        trajectory = []
        for i in range(len(t)):
            trajectory.append(
                CartesianTrajectoryPoint(
                    t=t[i],
                    X=Xd[i],
                    Xdot=Xd_dot[i],
                    Xddot=Xd_ddot[i]
                )
            )

        return trajectory

    # ----------------------------------------------------------
    #  FUNZIONI INTERNE
    # ----------------------------------------------------------

    def linear_cartesian_segment(self, X_start, X_end, dt, duration, Q_start=None, Q_end=None):
        """Genera una traiettoria cartesiana lineare con velocità costante."""
        t = np.arange(0, duration, dt)
        Xd = X_start + np.outer(t / duration, (X_end - X_start))
        Xd_dot = np.tile((X_end - X_start) / duration, (len(t), 1))
        Xd_ddot = np.zeros_like(Xd)
        traj = []

        if Q_start is not None:
            Qd = []
        for alpha in t / duration:
            Qd.append(self.slerp(Q_start, Q_end, alpha))
        else:
            Qd = [None] * len(t)

        for i in range(len(t)):
            traj.append(CartesianTrajectoryPoint(t[i], Xd[i], Xd_dot[i], Xd_ddot[i]))
        return traj


    def _compute_s(self, points):
        """
        Calcola l'ascissa curvilinea dei punti dati.

        :param points: array Nx3 di punti cartesiani
        """
        s = np.zeros(len(points))
        for i in range(1, len(points)):
            s[i] = s[i-1] + np.linalg.norm(points[i] - points[i-1])
        return s

    def slerp(self, q0, q1, alpha):
        r0 = tf.Rotation.from_quat(q0)
        r1 = tf.Rotation.from_quat(q1)
        return tf.Slerp([0,1], tf.Rotation.concatenate([r0, r1]))([alpha]).as_quat()[0]


    def _trapezoidal_time_scaling(self, L, vmax, amax):
        """
        Calcola il profilo di accellerazione trapezoidale.

        :param L: lunghezza del percorso
        :param vmax: velocità massima
        :param amax: accelerazione massima
        """
        t_acc = vmax / amax
        s_acc = 0.5 * amax * t_acc**2

        if 2*s_acc > L:
            t_acc = np.sqrt(L / amax)
            t_flat = 0
            s_acc = 0.5 * amax * t_acc**2
        else:
            s_flat = L - 2*s_acc
            t_flat = s_flat / vmax

        T = 2*t_acc + t_flat
        t = np.arange(0, T, self.dt)

        s_t = np.zeros_like(t)
        v_t = np.zeros_like(t)
        a_t = np.zeros_like(t)

        for i, ti in enumerate(t):
            if ti < t_acc:
                a_t[i] = amax
                v_t[i] = amax * ti
                s_t[i] = 0.5 * amax * ti**2
            elif ti < t_acc + t_flat:
                a_t[i] = 0
                v_t[i] = vmax
                s_t[i] = s_acc + vmax * (ti - t_acc)
            else:
                td = ti - (t_acc + t_flat)
                a_t[i] = -amax
                v_t[i] = vmax - amax * td
                s_t[i] = s_acc + t_flat*vmax + (vmax*td - 0.5*amax*td**2)

        return t, s_t, v_t, a_t

    def evaluate_trajectory(self, tck_x, tck_y, tck_z, s_t, v_t, a_t):
        """
        Calcola posizione, velocità e accelerazione lungo la traiettoria.
        
        :param tck_x: asse x della traiettoria
        :param tck_y: asse y della traiettoria
        :param tck_z: asse z della traiettoria
        :param s_t: ascissa curvilinea lungo la traiettoria
        :param v_t: velocità lungo la traiettoria
        :param a_t: accelerazione lungo la traiettoria
        """
        x = splev(s_t, tck_x, der=0)
        y = splev(s_t, tck_y, der=0)
        z = splev(s_t, tck_z, der=0)

        dx_ds = splev(s_t, tck_x, der=1)
        dy_ds = splev(s_t, tck_y, der=1)
        dz_ds = splev(s_t, tck_z, der=1)

        d2x_ds2 = splev(s_t, tck_x, der=2)
        d2y_ds2 = splev(s_t, tck_y, der=2)
        d2z_ds2 = splev(s_t, tck_z, der=2)

        vx = dx_ds * v_t
        vy = dy_ds * v_t
        vz = dz_ds * v_t

        ax = d2x_ds2 * (v_t**2) + dx_ds * a_t
        ay = d2y_ds2 * (v_t**2) + dy_ds * a_t
        az = d2z_ds2 * (v_t**2) + dz_ds * a_t

        Xd = np.vstack([x, y, z]).T
        Xd_dot = np.vstack([vx, vy, vz]).T
        Xd_ddot = np.vstack([ax, ay, az]).T

        '''
                # Inverti Y e Z per adattarti all'importazione Unity ruotata
        # ROS Z (altezza) -> Unity Y (altezza)
        # ROS Y (laterale) -> Unity Z (profondità)
        Xd = np.vstack([x, z, y]).T 
        Xd_dot = np.vstack([vx, vz, vy]).T
        Xd_ddot = np.vstack([ax, az, ay]).T
        '''
        return Xd, Xd_dot, Xd_ddot
        
def main():
    rclpy.init()
    node = TrajectoryManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()