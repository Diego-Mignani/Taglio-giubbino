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
        
        self.dt = 0.1
        self.received_waypoint = None
        self.trajectory_ready = False

        # 1. PARAMETRI E KDL
        self.declare_parameter('robot_description', '')
        self.robot_desc = self.get_parameter('robot_description').get_parameter_value().string_value
        print("DEBUG robot_description:", self.robot_desc)
        self.kinematics = kinematics.KDLKinematics6DOF(self.robot_desc)

        self.q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Da cambiare
        self.Q_down = tf.Rotation.from_euler('x', 180, degrees=True).as_quat()  #orientazione per andare giu
        
        self.T_home = self.kinematics.fk_6dof(self.q_home)  
        self.X_home = self.kinematics.position_from_T(self.T_home)
        self.R_home = self.kinematics.rotation_from_T(self.T_home)
        self.Q_home = self.kinematics.quaternion_from_R(self.R_home)   #90 0 0   offset x è 90
        
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
            self.X_home, X_first1, self.dt, duration=3,
            Q_start=self.Q_home, Q_end=self.Q_down
        )

        # -------------------------------
        # RETREAT (ultimo punto → home)
        # -------------------------------
        retreat_traj1 = self.linear_cartesian_segment(
            X_last1, self.X_home, self.dt, duration=3, 
            Q_start=self.Q_down, Q_end=self.Q_down
        )
        
        # -------------------------------
        # APPROACH (home → primo punto)
        # -------------------------------
        approach_traj2 = self.linear_cartesian_segment(
            self.X_home, X_first2, self.dt, duration=3, 
            Q_start=self.Q_down, Q_end=self.Q_down
        )

        # -------------------------------
        # RETREAT (ultimo punto → home)
        # -------------------------------
        retreat_traj2 = self.linear_cartesian_segment(
            X_last2, self.X_home, self.dt, duration=3, 
            Q_start=self.Q_down, Q_end=self.Q_home
        )

        # traiettoria totale: approach + tracking + retreat
        full_traj = approach_traj1 + self.tracking_traj1 + retreat_traj1 + approach_traj2 + self.tracking_traj2 + retreat_traj2
        
        # Riallina i timestamp cumulativi
        t_offset = 0.0
        for p in full_traj:
            p.t = p.t + t_offset
            if p.t > t_offset:
                t_offset = p.t


        # --- SERIALIZZAZIONE PER ROS 2 ---
        # Creiamo un'unica lista piatta: [t1, x1, y1, z1, vx1, vy1, vz1, ax1, ay1, az1, t2, x2, ...]
        flat_data = []
        for p in full_traj:
            flat_data.extend([p.t, *p.X, *p.Xdot, *p.Xddot,*p.Q])
            #flat_data.extend([p.t, *p.X, *p.Xdot, *p.Xddot])
        
        msg = Float64MultiArray()
        msg.data = flat_data
        self.pub_to_controller.publish(msg)
        self.get_logger().info(f'Inviata traiettoria di {len(full_traj)} punti.')
        self.trajectory_ready = True
        # Pubblica solo X,Y,Z per Unity
        flat_xyz = []
        for p in full_traj:
            flat_xyz.extend([p.X[0], p.X[1], p.X[2], *p.Q])

        msg_xyz = Float64MultiArray()
        msg_xyz.data = flat_xyz
        self.visualizza_unity.publish(msg_xyz)

    def plan_from_points(self, points_3d, vmax=0.6, amax=0.8, jmax=0.8, smooth=0.01):
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
        #t, s_t, v_t, a_t = self._trapezoidal_acceleration_profile(L, amax, jmax)

        # DEBUG: plot dei profili temporali
        #self.plot_time_scaling(t, s_t, v_t, a_t)

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
                    Xddot=Xd_ddot[i],
                    Q=self.Q_down
                )
            )

        return trajectory

    '''
    def plot_time_scaling(self, t, s_t, v_t, a_t):
        import matplotlib.pyplot as plt
        import numpy as np

        # jerk numerico
        j_t = np.gradient(a_t, t)

        plt.figure(figsize=(12, 10))

        plt.subplot(4, 1, 1)
        plt.plot(t, s_t, label="s(t)")
        plt.title("Ascissa curvilinea s(t)")
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(t, v_t, label="v(t)")
        plt.title("Velocità v(t)")
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(t, a_t, label="a(t)")
        plt.title("Accelerazione a(t)")
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(t, j_t, label="j(t)")
        plt.title("Jerk j(t)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    '''

    # ----------------------------------------------------------
    #  FUNZIONI INTERNE
    # ----------------------------------------------------------

    def linear_cartesian_segment(self, X_start, X_end, dt, duration, Q_start, Q_end):
        # 1) lunghezza del segmento
        L = np.linalg.norm(X_end - X_start)

        # 2) profilo trapezoidale in s(t)
        vmax = L / (duration / 2)   # stima semplice
        amax = vmax / 0.5           # accelera in metà tempo
        t, s_t, v_t, a_t = self._trapezoidal_time_scaling(L, vmax, amax)

        # 3) interpolazione lineare in posizione usando s(t)
        direction = (X_end - X_start) / L
        Xd = X_start + np.outer(s_t, direction)
        Xd_dot = np.outer(v_t, direction)
        Xd_ddot = np.outer(a_t, direction)

        # 4) interpolazione SLERP dell’orientazione
        traj = []
        for i in range(len(t)):
            alpha = s_t[i] / L
            Q_interp = self.slerp(Q_start, Q_end, alpha)
            traj.append(CartesianTrajectoryPoint(t[i], Xd[i], Xd_dot[i], Xd_ddot[i], Q=Q_interp))

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

    def _trapezoidal_acceleration_profile(self, L, amax, jmax):
        """
        Profilo trapezoidale in accelerazione (jerk costante).
        a(t) = trapezoidale
        v(t) = ∫ a(t)
        s(t) = ∫ v(t)

        :param L: lunghezza totale
        :param amax: accelerazione massima
        :param jmax: jerk massimo
        """

        # 1) Tempi delle fasi
        t_j = amax / jmax          # tempo per raggiungere amax
        t_a = t_j                  # simmetrico
        t_flat = 0                 # fase a accelerazione costante

        # 2) Spazio durante salita + discesa accelerazione
        # a sale linearmente → v è parabola → s è cubica
        s_j = (1/6) * jmax * t_j**3
        s_acc = 2 * s_j            # salita + discesa

        # 3) Se non basta per coprire L, aggiungi fase piatta
        if s_acc < L:
            s_flat = L - s_acc
            # velocità raggiunta al termine della salita
            v_peak = 0.5 * jmax * t_j**2
            if not np.isfinite(v_peak) or v_peak <= 0:
                raise ValueError(f"Invalid v_peak: {v_peak}")
            t_flat = s_flat / v_peak
            if not np.isfinite(t_flat) or t_flat < 0:
                raise ValueError(f"Invalid t_flat: {t_flat}")

        else:
            # profilo triangolare in accelerazione
            t_j = (L * 3 / jmax)**(1/3)
            t_a = t_j
            t_flat = 0

        # 4) Tempo totale
        T = 2*t_j + t_flat
        if not np.isfinite(T) or T <= 0:
            raise ValueError(f"Invalid total time T: {T}")

        t = np.arange(0, T, self.dt)

        # 5) Preallocazione
        a_t = np.zeros_like(t)
        v_t = np.zeros_like(t)
        s_t = np.zeros_like(t)

        # 6) Costruzione profilo
        for i, ti in enumerate(t):

            if ti < t_j:
                # fase 1: jerk positivo
                a = jmax * ti
                v = 0.5 * jmax * ti**2
                s = (1/6) * jmax * ti**3

            elif ti < t_j + t_flat:
                # fase 2: accelerazione costante
                dt = ti - t_j
                a = amax
                v = (0.5 * jmax * t_j**2) + amax * dt
                s = (1/6)*jmax*t_j**3 + (0.5*jmax*t_j**2)*dt + 0.5*amax*dt**2

            else:
                # fase 3: jerk negativo
                dt = ti - (t_j + t_flat)
                a = amax - jmax * dt
                v = (0.5*jmax*t_j**2 + amax*t_flat) + (amax*dt - 0.5*jmax*dt**2)
                s = (1/6)*jmax*t_j**3 + (0.5*jmax*t_j**2)*t_flat + 0.5*amax*t_flat**2 \
                    + (0.5*jmax*t_j**2)*dt + 0.5*amax*dt**2 - (1/6)*jmax*dt**3

            a_t[i] = a
            v_t[i] = v
            s_t[i] = s
            if not np.isfinite(s_t).all():
                raise ValueError("s_t contains invalid values")


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

        s_min = tck_x[0][0]
        s_max = tck_x[0][-1]
        s_t = np.clip(s_t, s_min, s_max)

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
