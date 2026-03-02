import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from tm5_pk_traiettorie.types import CartesianTrajectoryPoint
from sensor_msgs.msg import JointState
from rclpy.clock import Clock, ClockType
from std_msgs.msg import Bool
import numpy as np
from datetime import datetime
import csv
from .joint_controller import JointSpaceController6DOF
import os
import threading
import time

import numpy as np 

class RobotController(Node):
    def __init__(self):
        super().__init__('tm5_robot_controller')

        # Parametro URDF
        self.declare_parameter('robot_description', '')
        self.robot_desc = self.get_parameter('robot_description').get_parameter_value().string_value

        # Clock per misurare il tempo reale tra i callback
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)

        # Stato simulato del robot
        self.q_sim             = np.zeros(6)
        self.qdot_sim          = np.zeros(6)
        self.state_initialized = False

        # Log dei dt reali per analisi post-traiettoria
        self.dt_log = []
        self.t      = 0.0
        self.t_traj = 0.0

        # Variabile per indicare se la traiettoria è finita
        self.last_js_time = 0.0
        self.traj_finita  = False

        # Parametri di design del controller
        self.dt = 0.1  # 30Hz coerente con Unity (valore nominale)
        self.declare_parameter('Kp')
        self.declare_parameter('Kd')
        self.declare_parameter('K_ori')
        self.declare_parameter('w_ori')

        # Placeholder iniziali (veri valori arrivano dal YAML)
        self.Kp     = np.eye(6)
        self.Kd     = np.eye(6)
        self.K_ori  = 1.0
        self.w_ori  = 1.0

        self.param_timer = self.create_timer(0.1, self.load_params_once)

        self.controller = JointSpaceController6DOF(
            Kp=self.Kp,
            Kd=self.Kd,
            dt=self.dt,
            robot_desc=self.robot_desc,
            K_ori=self.K_ori,
            w_ori=self.w_ori
        )
        
        # Variabili per la traiettoria desiderata
        self.full_traj = []
        self.traiettoria_pronta = False

        # Stato corrente giunti
        self.current_js          = JointState()
        self.current_js.position = [0.0] * 6
        self.current_js.velocity = [0.0] * 6

        # NIENTE timer per il controllo: useremo un thread dedicato
        self.control_flag = False
        self.dt_real      = self.dt  # fallback

        # Vari canali di comunicazione
        self.publisher          = self.create_publisher(   Float64MultiArray, 'joint_commands',                                     10)
        self.joint_feedback_sub = self.create_subscription(JointState,        'unity_joint_feedback', self.joint_feedback_callback, 10)
        self.traj_sub           = self.create_subscription(Float64MultiArray, 'trajectory',           self.receive_trajectory,      10)
        self.ready_pub          = self.create_publisher(   Bool,              'robot_ready',                                        10)
        self.unity_ready_sub    = self.create_subscription(Bool,              'unity_ready',          self.unity_ready_callback,    10)

        # Handshake e stato di connessione
        self.unity_ready = False
        self.unity_connected = False

        # Thread di controllo dedicato
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
        self.get_logger().info("Thread di controllo avviato.")

        # Log dei dati per analisi post-traiettoria
        self.log_data = []
        self.prev_q = None
        self.prev_t = None


    # ------------------------------------------------------------------
    # CALLBACK E PARAMETRI
    # ------------------------------------------------------------------
    def unity_ready_callback(self, msg):
        if msg.data:
            self.get_logger().info("Unity ha segnalato di essere pronta.")
            self.unity_ready = True

    def load_params_once(self):
        """Carica i parametri dal YAML dopo che ROS2 li ha applicati."""
        Kp_list = self.get_parameter('Kp').value
        Kd_list = self.get_parameter('Kd').value
        K_ori_val = self.get_parameter('K_ori').value
        w_ori_val = self.get_parameter('w_ori').value

        if Kp_list is None or Kd_list is None or K_ori_val is None or w_ori_val is None:
            self.get_logger().warn("Parametri non ancora disponibili, ritento...")
            return

        self.Kp = np.diag(np.array(Kp_list, dtype=float))
        self.Kd = np.diag(np.array(Kd_list, dtype=float))
        self.K_ori = float(K_ori_val)
        self.w_ori = float(w_ori_val)

        self.get_logger().info(f"Parametri caricati dal YAML:")
        self.get_logger().info(f"Kp = {self.Kp}")
        self.get_logger().info(f"Kd = {self.Kd}")
        self.get_logger().info(f"K_ori = {self.K_ori}")
        self.get_logger().info(f"w_ori = {self.w_ori}")

        # Aggiorna il controller con i valori veri
        self.controller.Kp = self.Kp
        self.controller.Kd = self.Kd
        self.controller.K_ori = self.K_ori
        self.controller.w_ori = self.w_ori

        # Disattiva il timer
        self.param_timer.cancel()


    def joint_feedback_callback(self, msg):
        """
        Callback per ricevere il feedback dei giunti da Unity.
        """
        self.current_js = msg

        # Inizializza lo stato simulato alla configurazione di Unity
        if not self.state_initialized:
            self.q_sim = np.array(msg.position)

            if len(msg.velocity) == 6:
                self.qdot_sim = np.array(msg.velocity)
            else:
                self.qdot_sim = np.zeros(6)

            self.q_cmd = np.array(msg.position, dtype=float)
            self.state_initialized = True

        # Handshake di connessione con Unity
        if not self.unity_connected:
            self.get_logger().info("Handshake: Unity connesso! Segnalo al sistema.")
            self.unity_connected = True

        # Segnala che il robot è pronto
        ready_msg = Bool()
        ready_msg.data = True
        self.ready_pub.publish(ready_msg)

        # Calcolo dt_real
        now = self.steady_clock.now().nanoseconds * 1e-9

        if not hasattr(self, "prev_time"):
            self.prev_time = now

        raw_dt = now - self.prev_time
        self.prev_time = now

        # logghi il dt reale per statistiche
        self.dt_log.append(raw_dt)
        self.last_js_time = now

        # aggiorna dt_real usato dal controllo
        self.dt_real = raw_dt

        # alza il flag per il loop di controllo
        self.control_flag = True

    def receive_trajectory(self, msg):
        """
        Callback per ricevere la traiettoria desiderata.
        """
        self.traiettoria_pronta = True
        self.traj_finita = False
        self.dt_log = []

        # Tempo di inizio traiettoria (tempo assoluto ROS2)
        self.t0_traj = self.steady_clock.now().nanoseconds * 1e-9

        data = np.array(msg.data).reshape(-1, 14)
        self.full_traj = []

        for row in data:
            point = CartesianTrajectoryPoint(
                X=row[1:4],      # Posizione
                Xdot=row[4:7],   # Velocità
                Xddot=row[7:10], # Accelerazione
                t=row[0],
                Q=row[10:14]
            )
            self.full_traj.append(point)

        self.get_logger().info(f"Ricevuta traiettoria di {len(self.full_traj)} punti")


    # ------------------------------------------------------------------
    # LOOP DI CONTROLLO DEDICATO
    # ------------------------------------------------------------------
    def control_loop(self):
        """
        Loop di controllo "alla microcontrollore":
        gira in un thread separato, esegue il controllo solo quando
        control_flag è True e tutte le condizioni sono soddisfatte.
        """
        self.get_logger().info("control_loop avviato.")
        while rclpy.ok():
            if not self.control_flag:
                time.sleep(0.0005)
                continue

            # reset del flag (edge-triggered)
            self.control_flag = False

            # condizioni di sicurezza
            if (not self.unity_ready or
                not self.traiettoria_pronta or
                self.traj_finita or
                self.current_js is None or
                len(self.full_traj) == 0):
                continue

            try:
                self.control_step()
            except Exception as e:
                self.get_logger().error(f"Errore in control_step: {e}")

    # ------------------------------------------------------------------
    # SINGOLO STEP DI CONTROLLO (EX control_callback_sg1)
    # ------------------------------------------------------------------
    def control_step(self):
        """
        Un singolo step di controllo (ex control_callback_sg1),
        richiamato dal loop di controllo con dt_real.
        """

        # Fine traiettoria basata sul tempo assoluto
        now = self.steady_clock.now().nanoseconds * 1e-9
        t_real = now - self.t0_traj

        if t_real >= self.full_traj[-1].t and not self.traj_finita:

            self.get_logger().info("[DEBUG] Fine traiettoria raggiunta (tempo)")

            self.traj_finita = True
            dt_array = np.array(self.dt_log, dtype=float)
            dt_mean = np.mean(self.dt_log)
            dt_max = np.max(self.dt_log)
            dt_min = np.min(self.dt_log)
            conteggio = np.sum(dt_array > 0.12)

            IAE, ISE, ITAE, RMSE = self.controller.compute_performance_indices(dt_mean)

            self.get_logger().info("=== INDICI DI PRESTAZIONE ===")
            self.get_logger().info(f"IAE  = {IAE:.6f}")
            self.get_logger().info(f"ISE  = {ISE:.6f}")
            self.get_logger().info(f"ITAE = {ITAE:.6f}")
            self.get_logger().info(f"RMSE = {RMSE:.6f}")
            self.get_logger().info("==============================")

            self.salva_dati(IAE, ISE, ITAE, RMSE, dt_mean, dt_max, dt_min, conteggio)
            self.salva_log_traiettoria()
            self.get_logger().info(
                f"dt_real — media: {dt_mean:.6f}, max: {dt_max:.6f}, min: {dt_min:.6f}"
            )
            return


        # Se Unity non è ancora connesso, non calcolare nulla
        if not self.unity_connected or self.current_js is None:
            return

        # 1. Traiettoria desiderata
        Xd, Xd_dot, Xd_ddot, Qd = self.evaluate()

        now = self.steady_clock.now().nanoseconds * 1e-9
        q = np.array(self.current_js.position)

        if self.prev_q is None:
            q_dot_real = np.zeros(6)
        else:
            dt = now - self.prev_t
            q_dot_real = (q - self.prev_q) / max(dt, 1e-6)

        self.prev_q = q.copy()
        self.prev_t = now

        # Sovrascrivi la velocity del JointState
        self.current_js.velocity = q_dot_real.tolist()


        # 2. Comando dal controller
        qdot_cmd, X, Xdot_real, Q_act, omega_real = self.controller.compute_command_operational(
            self.current_js,
            Xd, Xd_dot, Xd_ddot, Qd
        )

        # 3. Stato attuale
        q = np.array(self.current_js.position)
        qdot = np.array(self.current_js.velocity) if len(self.current_js.velocity) == 6 else np.zeros(6)

        # 4. Integrazione dinamica con dt_real
        dt = float(self.dt_real) if hasattr(self, "dt_real") else self.dt
        dt = max(0.0005, min(dt, 0.2))  # clamp di sicurezza

        q_new = q + qdot_cmd * dt

        # 5. Pubblica verso Unity
        msg = Float64MultiArray()
        msg.data = q_new.tolist()
        self.publisher.publish(msg)

        self.log_data.append({
            "t":        t_real,
            "Xd":       Xd.copy(),
            "Xd_dot":   Xd_dot.copy(),
            "Xd_ddot":  Xd_ddot.copy(),
            "X":        X.copy(),
            "Xdot":     Xdot_real.copy(),
            "Qd":       Qd.copy(),
            "Q":        Q_act.copy(),
            "omega":    omega_real.copy(),
        })
            

    def evaluate(self):
        """
        Valuta la traiettoria usando il tempo assoluto ROS2.
        Interpola posizione, velocità e accelerazione.
        Orientazione: prende il punto più vicino (stabile).
        """

        # tempo reale dall'inizio della traiettoria
        now = self.steady_clock.now().nanoseconds * 1e-9
        t_real = now - self.t0_traj

        times = np.array([p.t for p in self.full_traj])

        # prima dell'inizio
        if t_real <= times[0]:
            p = self.full_traj[0]
            return p.X, p.Xdot, p.Xddot, p.Q

        # dopo la fine
        if t_real >= times[-1]:
            p = self.full_traj[-1]
            return p.X, p.Xdot, p.Xddot, p.Q

        # trova il primo timestamp >= t_real
        i = np.searchsorted(times, t_real)

        p0 = self.full_traj[i-1]
        p1 = self.full_traj[i]

        t0 = p0.t
        t1 = p1.t

        # coefficiente di interpolazione
        alpha = (t_real - t0) / (t1 - t0)
        alpha = np.clip(alpha, 0.0, 1.0)

        # interpolazione lineare posizione/velocità/accelerazione
        Xd = p0.X + alpha * (p1.X - p0.X)
        Xd_dot = p0.Xdot + alpha * (p1.Xdot - p0.Xdot)
        Xd_ddot = p0.Xddot + alpha * (p1.Xddot - p0.Xddot)

        # orientazione: prendi quella più vicina (stabile)
        Qd = p0.Q if alpha < 0.5 else p1.Q

        return Xd, Xd_dot, Xd_ddot, Qd

    def salva_log_traiettoria(self):
        save_dir = os.path.expanduser("~/ros2_ws/log_traj")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "traj_log.csv")

        import csv
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t",
                "Xd_x","Xd_y","Xd_z",
                "Xd_dot_x","Xd_dot_y","Xd_dot_z",
                "Xd_ddot_x","Xd_ddot_y","Xd_ddot_z",
                "X_x","X_y","X_z",
                "Xdot_x","Xdot_y","Xdot_z",
                "Qd_x","Qd_y","Qd_z","Qd_w",
                "Q_x","Q_y","Q_z","Q_w",
                "omega_x","omega_y","omega_z"
            ])

            for row in self.log_data:
                writer.writerow([
                    row["t"],
                    *row["Xd"],
                    *row["Xd_dot"],
                    *row["Xd_ddot"],
                    *row["X"],
                    *row["Xdot"],
                    *row["Qd"],
                    *row["Q"],
                    *row["omega"]
                ])

    # ------------------------------------------------------------------
    # SALVATAGGIO PRESTAZIONI
    # ------------------------------------------------------------------
    def salva_dati(self, IAE, ISE, ITAE, RMSE, dt_mean, dt_max, dt_min, conteggio):
        save_dir = os.path.expanduser("~/ros2_ws/prestazioni")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "prestazioni_robot.csv")

        write_header = not os.path.exists(file_path)

        Kp_str = ",".join(map(str, np.diag(self.Kp).tolist()))
        Kd_str = ",".join(map(str, np.diag(self.Kd).tolist()))

        with open(file_path, mode="a", newline="") as f:
            writer = csv.writer(f, delimiter=';')
            if write_header:
                writer.writerow([
                    "timestamp", "IAE", "ISE", "ITAE", "RMSE",
                    "Kp", "Kd", "K_ori", "w_ori",
                    "traj_len", "dt_mean", "dt_max", "dt_min", "conteggio_dt>0.12"
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{IAE:.6f}".replace('.', ','),
                f"{ISE:.6f}".replace('.', ','),
                f"{ITAE:.6f}".replace('.', ','),
                f"{RMSE:.6f}".replace('.', ','),
                Kp_str,
                Kd_str,
                self.K_ori,
                self.w_ori,
                len(self.full_traj),
                dt_mean,
                dt_max,
                dt_min,
                conteggio
            ])

def main():
    rclpy.init()
    node = RobotController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
