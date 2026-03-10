from sensor_msgs.msg import JointState
from my_robot_utils import kinematics
from scipy.spatial.transform import Rotation as R
import numpy as np

def compute_angular_velocity(Q_prev, Q_curr, dt):
    """
    Q_prev, Q_curr: [x, y, z, w]
    """
    Q_prev = np.array(Q_prev, dtype=float)
    Q_curr = np.array(Q_curr, dtype=float)

    R_prev = R.from_quat(Q_prev)
    R_curr = R.from_quat(Q_curr)

    R_rel = R_curr * R_prev.inv()
    rotvec = R_rel.as_rotvec()  # asse * angolo

    if dt <= 0:
        return np.zeros(3)

    return rotvec / dt


class JointSpaceController6DOF:
    def __init__(self, Kp: np.ndarray, Kd: np.ndarray, dt: float, robot_desc: str, K_ori=1.0, w_ori=1.0):
        """
        Kp, Kd: matrici 6x6 (tipicamente diagonali)
        """
        self.Kp = Kp
        self.Kd = Kd
        self.dt = dt
        self.qd_dot_prev = np.zeros(6)
        self.kinematics = kinematics.KDLKinematics6DOF(robot_desc)
        self.error_history = []
        self.K_ori = K_ori 
        self.w_ori = w_ori

         # nel __init__
        self.qd_prev = np.zeros(6)
        self.qd_dot_prev_filt = np.zeros(6)
        self.alpha_qd = 0.2      # filtro su posizione desiderata
        self.alpha_qd_dot = 0.2  # filtro su velocità desiderata

    def compute_command_joint(self, q, q_dot, qd, qd_dot):
        """
        Controllo PD in spazio dei giunti:
        q, q_dot: stato attuale
        qd, qd_dot: riferimento da MoveIt2
        Ritorna: qdot_cmd
        """
        e = qd - q
        e_dot = qd_dot - q_dot

        # Kp, Kd sono 6x6
        qdot_cmd = qd_dot + self.Kp @ e + self.Kd @ e_dot

        # saturazione
        max_qdot = 2.0
        qdot_cmd = np.clip(qdot_cmd, -max_qdot, max_qdot)

        return qdot_cmd



    def compute_performance_indices(self, dt):
        """
        Performance indices:
        - IAE: ∫|e(t)| dt
        - ISE: ∫e(t)² dt
        - ITAE: ∫t|e(t)| dt
        - RMSE: sqrt(mean(e(t)²))
        """
        e = np.array(self.error_history)
        IAE = np.sum(np.abs(e)) * dt
        ISE = np.sum(e**2) * dt
        ITAE = np.sum(np.arange(len(e)) * dt * np.abs(e))
        RMSE = np.sqrt(np.mean(e**2))
        return IAE, ISE, ITAE, RMSE
    
    def orientation_error_angle(self, Qd, Q_act):
        """ Funzione di errore orientazione come angolo equivalente (radiani) """
        dot = np.dot(Qd, Q_act)
        dot = np.clip(dot, -1.0, 1.0)
        return 2 * np.arccos(abs(dot))   # radianti

