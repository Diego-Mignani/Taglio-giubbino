import numpy as np
from tm5_pk_traiettorie.types import CartesianTrajectoryPoint
from sensor_msgs.msg import JointState
from my_robot_utils import kinematics

class JointSpaceController6DOF:
    def __init__(self, Kp: np.ndarray, Kd: np.ndarray, dt: float, robot_desc: str):
        """
        Kp, Kd: matrici 6x6 (tipicamente diagonali)
        """
        self.Kp = Kp
        self.Kd = Kd
        self.dt = dt
        self.qd_dot_prev = np.zeros(6)
        self.kinematics = kinematics.KDLKinematics6DOF(robot_desc)
        self.error_history = []

         # nel __init__
        self.qd_prev = np.zeros(6)
        self.qd_dot_prev_filt = np.zeros(6)
        self.alpha_qd = 0.2      # filtro su posizione desiderata
        self.alpha_qd_dot = 0.2  # filtro su velocità desiderata


    #---------------------------------------------------------
    #    CONTROLLO ALESSANDRO SPAZIO DEI GIUNTI
    #--------------------------------------------------------
    def compute_command(self, js: JointState, Xd, Xd_dot, Xd_ddot):
        q = np.array(js.position)
        q_dot = np.array(js.velocity)
        if len(q_dot) == 0:
            q_dot = np.zeros_like(q)

        # 1. IK -> qd
        #qd = self.kinematics.ik_position(Xd, q_seed=q)
        
        # 1. IK -> qd (grezzo)
        qd_raw = self.kinematics.ik_position(Xd, q_seed=q)

         # Filtro su qd per evitare salti dall'IK 
        if not hasattr(self, "qd_prev"): 
            self.qd_prev = qd_raw.copy()

        # filtro semplice per evitare salti da IK
        qd = self.qd_prev + self.alpha_qd * (qd_raw - self.qd_prev)
        self.qd_prev = qd.copy()

        # 2. J(qd) -> qd_dot (grezzo)
        J_full = self.kinematics.get_full_jacobian(qd)
        J_pos = J_full[:3, :]

        U, S, Vt = np.linalg.svd(J_pos, full_matrices=False)

        lambda2 = 1e-4  # damping
        S_damped = S / (S**2 + lambda2)
        J_pos_pinv = (Vt.T * S_damped) @ U.T

        qd_dot_raw = J_pos_pinv @ Xd_dot

        # Limita le velocità desiderate
        max_qd_dot = 1.5  # rad/s, da tarare
        qd_dot_raw = np.clip(qd_dot_raw, -max_qd_dot, max_qd_dot)

         # Filtro su qd_dot 
        if not hasattr(self, "qd_dot_prev_filt"): 
            self.qd_dot_prev_filt = qd_dot_raw.copy()

        # filtro su velocità desiderata
        qd_dot = self.qd_dot_prev_filt + self.alpha_qd_dot * (qd_dot_raw - self.qd_dot_prev_filt)
        self.qd_dot_prev_filt = qd_dot.copy()

        # 3. qd_ddot numerica
        qd_ddot = (qd_dot - self.qd_dot_prev) / self.dt
        self.qd_dot_prev = qd_dot.copy()

        max_qd_ddot = 10.0  # rad/s^2, da tarare
        qd_ddot = np.clip(qd_ddot, -max_qd_ddot, max_qd_ddot)


        e_q = qd - q
        e_qdot = qd_dot - q_dot
        #u = qd_ddot + self.Kp @ e_q + self.Kd @ e_qdot
        u = self.Kp @ e_q + self.Kd @ e_qdot


        error_norm = np.linalg.norm(e_q)
        self.error_history.append(error_norm)

    
        return u, qd, qd_dot, qd_ddot



    #---------------------------------------------------------
    #    CONTROLLO ALESSANDRO SPAZIO DEI GIUNTI POSIZIONE
    #--------------------------------------------------------
    def compute_command2(self, js: JointState, traj_point: CartesianTrajectoryPoint):
        """
        Controllo cinematico PD per robot industriali.
        Input:
            js          -> JointState reale (Unity o robot)
            traj_point  -> punto cartesiano desiderato (Xd, Xd_dot)
        Output:
            q_cmd       -> posizione giunto da inviare al robot
            qd          -> posizione desiderata (IK)
            qd_dot      -> velocità desiderata (Jacobian)
        """

        # --- 1) Stato reale del robot ---
        q = np.array(js.position)
        q_dot = np.array(js.velocity) if len(js.velocity) == 6 else np.zeros(6)

        # --- 2) Traiettoria cartesiana desiderata ---
        Xd = traj_point.X
        Xd_dot = traj_point.Xdot

        # --- 3) IK: posizione giunti desiderata ---
        qd = self.kinematics.ik_position(Xd, q_seed=q)

        # --- 4) Jacobiano per ottenere qd_dot ---
        J = self.kinematics.get_full_jacobian(qd)
        J_pos = J[:3, :]                      # solo parte lineare
        J_pinv = np.linalg.pinv(J_pos, rcond=1e-2)
        qd_dot = J_pinv @ Xd_dot              # velocità giunti desiderata

        # --- 5) Errore sui giunti ---
        e_q = qd - q
        e_qdot = qd_dot - q_dot

        # --- 6) CONTROLLO CINEMATICO PD ---
        q_cmd = qd + self.Kp @ e_q + self.Kd @ e_qdot

        return q_cmd, qd, qd_dot


    #---------------------------------------------------------
    #    CONTROLLO ALESSANDRO SPAZIO OPERATIVO FEEDBACK GIUNTI
    #--------------------------------------------------------
    def compute_command_so(self, js: JointState, traj_point: CartesianTrajectoryPoint):
        """
        Controllo in spazio operativo (cartesiano) con comando in velocità dei giunti.
        Funziona molto meglio in Unity rispetto al controllo in giunto.
        """

        # --- Stato reale ---
        q = np.array(js.position)
        q_dot = np.array(js.velocity) if len(js.velocity) == 6 else np.zeros(6)

        # --- Cinematica diretta ---
        T = self.kinematics.fk_6dof(q)
        X = self.kinematics.position_from_T(T)

        # --- Jacobiano ---
        J = self.kinematics.get_full_jacobian(q)
        J_pos = J[:3, :]   # parte lineare

        # --- Traiettoria desiderata ---
        Xd = traj_point.X
        Xd_dot = traj_point.Xdot

        # --- Errori cartesiani ---
        e_x = Xd - X
        e_xdot = Xd_dot - (J_pos @ q_dot)

        # --- Controllo cartesiano ---
        # Xdot_cmd = Xd_dot + Kp*e_x + Kd*e_xdot
        Xdot_cmd = Xd_dot + self.Kp[:3,:3] @ e_x + self.Kd[:3,:3] @ e_xdot

        # --- Converti in velocità giunti ---
        J_pinv = np.linalg.pinv(J_pos, rcond=1e-2)
        qdot_cmd = J_pinv @ Xdot_cmd

        return qdot_cmd



    #---------------------------------------------------------
    #    CONTROLLO ALESSANDRO SPAZIO OPERATIVO FEEDBACK POSA EE
    #--------------------------------------------------------
    def compute_command_so2(self,
                        js: JointState,
                        X_real: np.ndarray,
                        traj_point: CartesianTrajectoryPoint):
        """
        Controllo in spazio operativo con feedback cartesiano reale.

        js      : JointState reale (per q e qdot e Jacobiano)
        X_real  : posizione cartesiana reale dell'EE (Unity)
        traj_point: Xd, Xd_dot della traiettoria
        """

        # Stato reale nei giunti
        q = np.array(js.position)
        q_dot = np.array(js.velocity) if len(js.velocity) == 6 else np.zeros(6)

        # Posizione cartesiana reale: la prendi da Unity, NON rifai FK
        X = np.array(X_real, dtype=float)

        # Jacobiano nel punto attuale (dai giunti)
        J = self.kinematics.get_full_jacobian(q)
        J_pos = J[:3, :]

        # Traiettoria desiderata
        Xd = traj_point.X
        Xd_dot = traj_point.Xdot

        # Errori cartesiani
        e_x = Xd - X
        Xdot_real = J_pos @ q_dot
        e_xdot = Xd_dot - Xdot_real
        
    
        error_norm = np.linalg.norm(e_x)
        self.error_history.append(error_norm)
        
        # Controllo cartesiano (uso solo i 3x3 di testa di Kp/Kd)
        Kp_pos = self.Kp[:3, :3]
        Kd_pos = self.Kd[:3, :3]

        Xdot_cmd = Xd_dot + Kp_pos @ e_x + Kd_pos @ e_xdot

        # Converti in velocità giunti
        J_pinv = np.linalg.pinv(J_pos, rcond=1e-2)
        qdot_cmd = J_pinv @ Xdot_cmd

        return qdot_cmd

    def compute_performance_indices(self, dt):
        e = np.array(self.error_history)

        IAE = np.sum(np.abs(e)) * dt
        ISE = np.sum(e**2) * dt
        ITAE = np.sum(np.arange(len(e)) * dt * np.abs(e))
        RMSE = np.sqrt(np.mean(e**2))

        return IAE, ISE, ITAE, RMSE
