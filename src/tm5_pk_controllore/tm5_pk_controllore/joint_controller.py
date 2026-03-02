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

    def compute_command_operational(self, js, Xd, Xd_dot, Xd_ddot, Qd):
        """
        Controllo in spazio operativo (posizione + velocità + accelerazione)
        con Jacobiano smorzato e comando in velocità dei giunti.
        Compatibile con Unity (nessun modello dinamico richiesto).
        """

        # --- Stato reale ---
        q = np.array(js.position)
        q_dot = np.array(js.velocity) if len(js.velocity) == 6 else np.zeros(6)

        # --- Cinematica diretta ---
        T = self.kinematics.fk_6dof(q)
        X = self.kinematics.position_from_T(T)
        R_act = self.kinematics.rotation_from_T(T)
        Q_act = self.kinematics.quaternion_from_R(R_act)

        # --- Jacobiano ---
        J = self.kinematics.get_full_jacobian(q)
        J_pos = J[:3, :]      # parte lineare
        J_ori = J[3:6, :]     # parte angolare

        # --- Errori cartesiani ---
        e_pos = Xd - X
        e_ori = self.kinematics.quat_error(Qd, Q_act)

        # errore orientazione come angolo equivalente
        err_ori = self.orientation_error_angle(Qd, Q_act)

        # errore totale combinato (posizione + orientazione)
        err_total = np.linalg.norm(e_pos) + err_ori

        self.error_history.append(err_total)

        # --- Velocità cartesiana reale ---
        Xdot_real = J_pos @ q_dot
        omega_real = J_ori @ q_dot

        e_vel = Xd_dot - Xdot_real
        e_omega = omega_d = Xd_ddot*0  # placeholder se vuoi aggiungere orientazione dinamica

        # --- Controllo posizione ---
        Kp_pos = self.Kp[:3, :3]
        Kd_pos = self.Kd[:3, :3]
        Xddot_cmd = Xd_ddot + Kp_pos @ e_pos + Kd_pos @ e_vel

        # --- Controllo orientazione ---
        Kp_ori = self.K_ori * np.eye(3)
        Kd_ori = self.w_ori * np.eye(3)
        omega_des = np.zeros(3)
        omega_err = omega_des - omega_real
        omega_cmd = Kp_ori @ e_ori + Kd_ori @ omega_err

        # --- Comando cartesiano completo ---
        V_cmd = np.hstack([Xddot_cmd, omega_cmd])

        # --- Jacobiano completo (damped least squares) ---
        lambda2 = 1e-4
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda2 * np.eye(6))

        qdot_cmd = q_dot + J_pinv @ (V_cmd * self.dt)

        # --- Saturazione ---
        max_qdot = 2.0
        qdot_cmd = np.clip(qdot_cmd, -max_qdot, max_qdot)

        return qdot_cmd, X, Xdot_real, Q_act, omega_real


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

