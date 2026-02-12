import numpy as np
from tm5_pk_traiettorie.types import CartesianTrajectoryPoint
from sensor_msgs.msg import JointState
from my_robot_utils import kinematics
from scipy.spatial.transform import Rotation as R


from scipy.spatial.transform import Rotation as R

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


    def compute_command(self, js: JointState, Xd, Xd_dot, Xd_ddot, Qd):
        q = np.array(js.position)
        q_dot = np.array(js.velocity)
        if len(q_dot) == 0:
            q_dot = np.zeros_like(q)

        # === FK: stato attuale ===
        T = self.kinematics.fk_6dof(q)
        X_actual = self.kinematics.position_from_T(T)
        R_act = self.kinematics.rotation_from_T(T)
        q_actual = self.kinematics.quaternion_from_R(R_act)

        # === Errori cartesiani ===
        e_pos = Xd - X_actual
        e_ori = self.kinematics.quat_error(Qd, q_actual)
        e_ori = self.w_ori * e_ori
        e_ori = np.clip(e_ori, -0.6, 0.6)

        # === 1) IK SOLO POSIZIONE ===
        qd_raw = self.kinematics.ik_position(Xd, q_seed=q)

        # === 1b) CORREZIONE ORIENTAZIONE SOPRA qd_raw ===
        J_full_qd = self.kinematics.get_full_jacobian(qd_raw)
        J_omega = J_full_qd[3:6, :]

        delta_q_ori = self.K_ori * np.linalg.pinv(J_omega) @ e_ori
        qd_raw += delta_q_ori

        # === filtro su qd ===
        if not hasattr(self, "qd_prev"):
            self.qd_prev = qd_raw.copy()

        qd = self.qd_prev + self.alpha_qd * (qd_raw - self.qd_prev)
        self.qd_prev = qd.copy()

        # === 2) velocità desiderata (posizione + orientazione) ===
        J_full = self.kinematics.get_full_jacobian(qd)

        if not hasattr(self, "Qd_prev"):
            self.Qd_prev = Qd.copy()

        omega_d = compute_angular_velocity(self.Qd_prev, Qd, self.dt)
        self.Qd_prev = Qd.copy()

        Vd = np.hstack([Xd_dot, omega_d])   # [vx, vy, vz, wx, wy, wz]
        qd_dot_raw = np.linalg.pinv(J_full) @ Vd

        max_qd_dot = 3.0
        qd_dot_raw = np.clip(qd_dot_raw, -max_qd_dot, max_qd_dot)

        # === filtro su qd_dot ===
        if not hasattr(self, "qd_dot_prev_filt"):
            self.qd_dot_prev_filt = qd_dot_raw.copy()

        qd_dot = self.qd_dot_prev_filt + self.alpha_qd_dot * (qd_dot_raw - self.qd_dot_prev_filt)
        self.qd_dot_prev_filt = qd_dot.copy()

        # === accelerazione desiderata ===
        qd_ddot = (qd_dot - self.qd_dot_prev) / self.dt
        self.qd_dot_prev = qd_dot.copy()
        qd_ddot = np.clip(qd_ddot, -20, 20)

        # === PD in giunto ===
        e_q = qd - q
        e_qdot = qd_dot - q_dot
        u = qd_ddot + self.Kp @ e_q + self.Kd @ e_qdot

        self.error_history.append(np.linalg.norm(e_q))

        return u, qd, qd_dot, qd_ddot

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

        # --- Velocità cartesiana reale ---
        Xdot_real = J_pos @ q_dot
        omega_real = J_ori @ q_dot

        e_vel = Xd_dot - Xdot_real
        e_omega = omega_d = Xd_ddot*0  # placeholder se vuoi aggiungere orientazione dinamica

        # --- Controllo cartesiano (PD + feedforward) ---
        Kp = self.Kp[:3, :3]
        Kd = self.Kd[:3, :3]

        Xddot_cmd = Xd_ddot + Kp @ e_pos + Kd @ e_vel

        # --- Converti accelerazione cartesiana in velocità giunti ---
        # Jacobiano smorzato (damped least squares)
        lambda2 = 1e-4
        J_pos_pinv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda2 * np.eye(3))

        qdot_cmd = q_dot + J_pos_pinv @ (Xddot_cmd * self.dt)

        # --- Filtro e saturazione ---
        max_qdot = 2.0
        qdot_cmd = np.clip(qdot_cmd, -max_qdot, max_qdot)

        
        self.error_history.append(np.linalg.norm(e_pos))

        return qdot_cmd, X, Xdot_real, Q_act, omega_real


    def compute_performance_indices(self, dt):
        e = np.array(self.error_history)

        IAE = np.sum(np.abs(e)) * dt
        ISE = np.sum(e**2) * dt
        ITAE = np.sum(np.arange(len(e)) * dt * np.abs(e))
        RMSE = np.sqrt(np.mean(e**2))

        return IAE, ISE, ITAE, RMSE
