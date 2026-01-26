from datetime import datetime
import csv
from scipy.interpolate import splrep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from tm5_pk_traiettorie.types import CartesianTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np
from std_msgs.msg import Bool 
from geometry_msgs.msg import Pose
from .joint_controller import JointSpaceController6DOF
from my_robot_utils import kinematics
import pandas as pd 
import os

class RobotController(Node):
    def __init__(self):
        super().__init__('tm5_robot_controller')

        # Stato simulato del robot
        self.q_sim = np.zeros(6)
        self.qdot_sim = np.zeros(6)
        self.state_initialized = False

        self.dt_log =[]
        self.t = 0.0
        self.t_traj = 0.0

        self.last_js_time =0
        self.traj_finita = False


        # Parametro URDF
        self.declare_parameter('robot_description', '')
        self.robot_desc = self.get_parameter('robot_description').get_parameter_value().string_value

        # Parametri di design del controller
        self.dt = 0.1 # 30Hz coerente con Unity
        #self.dt = 0.0333
    
        #self.Kp = np.diag([135,100,135,110,85,1])
        #self.Kd = np.diag([5,15,5,10,6,6])

        #self.Kp = np.diag([105,100,105,100,85,1])
        #self.Kd = np.diag([5,15,5,10,6,6])

        self.declare_parameter('Kp')
        self.declare_parameter('Kd')
        self.declare_parameter('K_ori')
        self.declare_parameter('w_ori')

        # Placeholder iniziali (veri valori arrivano dal YAML) 
        self.Kp = np.eye(6) 
        self.Kd = np.eye(6) 
        self.K_ori = 1.0 
        self.w_ori = 1.0

        self.param_timer = self.create_timer(0.1, self.load_params_once)




        self.controller = JointSpaceController6DOF(
            Kp=self.Kp,
            Kd=self.Kd,
            dt=self.dt,
            robot_desc=self.robot_desc,
            K_ori=self.K_ori,
            w_ori=self.w_ori
        )


        self.traj_duration = 0.0

        # Variabili di stato del controller
        self.q_dot_cmd = np.zeros(6)
        self.q_cmd = np.zeros(6, dtype=float)

        # Inizializza il controller e la cinematica
        #self.controller = JointSpaceController6DOF(self.Kp, self.Kd, self.dt, self.robot_desc)
        self.kinematics = kinematics.KDLKinematics6DOF(self.robot_desc)
        self.num_joints = self.kinematics.num_joints
        
        # Variabili per la traiettoria desiderata
        self.traj_index = 0      # Indice della traiettoria
        self.full_traj = []      
        self.traiettoria_pronta = False

        # Variabili per il feedback della posa end-effector da Unity
        self.posizione_reale_unity = np.zeros(3)
        self.orientazione_reale_unity =  np.zeros(4)
        self.ee_pose_valid = False

        # Variabile per lo stato corrente dei giunti
        self.current_js = JointState()
        self.current_js.position = [0.0] * 6
        self.current_js.velocity = [0.0] * 6

        # Timer per il controllo
        self.timer = self.create_timer(self.dt, self.control_callback_sg1)
        #self.timer = self.create_timer(self.dt, self.control_callback_so1)

        # Vari canali di comunicazione
        self.publisher = self.create_publisher(Float64MultiArray, 'joint_commands', 10)
        self.pose_ee_subscription = self.create_subscription(Pose, 'unity_end_effector_pose', self.pose_ee_feedback_callback, 10)
        self.joint_feedback_sub = self.create_subscription(JointState, 'unity_joint_feedback', self.joint_feedback_callback, 10)
        self.traj_sub = self.create_subscription(Float64MultiArray, 'trajectory', self.receive_trajectory, 10)
        self.ready_pub = self.create_publisher(Bool, 'robot_ready', 10)
        self.unity_ready = False
        self.unity_ready_sub = self.create_subscription(Bool,'unity_ready',self.unity_ready_callback, 10)
        self.unity_connected = False
    
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


    def pose_ee_feedback_callback(self, msg):
        '''
        Callback per ricevere la posa reale dell'end-effector da Unity.
        Args:
            msg (Pose): Messaggio contenente la posa dell'end-effector
        '''
        self.posizione_reale_unity = np.array( [msg.position.x, msg.position.y, msg.position.z], dtype=float )
        self.orientazione_reale_unity = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.ee_pose_valid = True

    
    def joint_feedback_callback(self, msg):
        '''
        Callback per ricevere il feedback dei giunti da Unity.
        Args:
            msg (JointState): Messaggio contenente lo stato dei giunti
        '''
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
        now = self.get_clock().now().nanoseconds * 1e-9
        # Crea t0_traj SOLO quando tutto è pronto 
        if (not hasattr(self, "t0_traj") 
            and self.traiettoria_pronta 
            and self.unity_connected 
            and self.current_js is not None):

            self.t0_traj = now
            self.get_logger().info(f"[DEBUG] t0_traj creato = {self.t0_traj:.3f}")
            return

        # Se t0_traj non è ancora definito, non fare controllo 
        if not hasattr(self, "t0_traj"): return 
        
        # Calcolo dt_real 
        now = self.get_clock().now().nanoseconds * 1e-9 
        if not hasattr(self, "prev_time"): self.prev_time = now 
        
        raw_dt = now - self.prev_time
        self.prev_time = now

        # logghi il dt reale per statistiche
        self.dt_log.append(raw_dt)
        self.last_js_time = self.get_clock().now().nanoseconds * 1e-9
    


    def receive_trajectory(self, msg):
        '''
        Callback per ricevere la traiettoria desiderata.
        Args:
            msg (Float64MultiArray): Messaggio contenente la traiettoria 
        '''

        self.traj_index = 0
        self.t_traj = 0.0
 
        self.traiettoria_pronta = True 
        self.traj_finita = False 
        self.t = 0.0 
        self.dt_log = []


        if hasattr(self, "t0_traj"): 
            del self.t0_traj 
        if hasattr(self, "prev_time"): 
            del self.prev_time 
        # Ipotizziamo che msg.data sia un array piatto di N punti * 10 valori
        data = np.array(msg.data).reshape(-1, 14)
        self.full_traj = []
        
        for row in data:
            # row: [t, x, y, z, vx, vy, vz, ax, ay, az]
            point = CartesianTrajectoryPoint(
                X=row[1:4],      # Posizione
                Xdot=row[4:7],   # Velocità
                Xddot=row[7:10], # Accelerazione
                t=row[0],
                Q = row[10:14] 
            )
            self.full_traj.append(point)
        
        self.traj_index = 0
        self.traiettoria_pronta = True
        self.get_logger().info(f"Ricevuta traiettoria di {len(self.full_traj)} punti")
        self.traj_duration = (len(self.full_traj) - 1) * self.dt


    def salva_dati(self, IAE, ISE, ITAE, RMSE, dt_mean, dt_max, dt_min, conteggio):
        save_dir = os.path.expanduser("~/ros2_ws/prestazioni")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "prestazioni_robot.csv")

        write_header = not os.path.exists(file_path)

        Kp_str = ",".join(map(str, np.diag(self.Kp).tolist()))
        Kd_str = ",".join(map(str, np.diag(self.Kd).tolist()))

        with open(file_path, mode="a", newline="") as f:
            writer = csv.writer(f, delimiter=';')  # separatore di colonna corretto
            if write_header:
                writer.writerow(["timestamp", "IAE", "ISE", "ITAE", "RMSE", "Kp", "Kd", "traj_len","dt_mean","dt_max","dt_min"])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{IAE:.6f}".replace('.', ','),   # opzionale: decimali in formato italiano
                f"{ISE:.6f}".replace('.', ','),
                f"{ITAE:.6f}".replace('.', ','),
                f"{RMSE:.6f}".replace('.', ','),
                Kp_str,
                Kd_str,
                len(self.full_traj),
                dt_mean,
                dt_max,
                dt_min,
                conteggio
            ])
    
    #-----------------------------------------------------------------
    #   CONTROLLO SPAZIO DEI GIUNTI DI ALESSANDRO
    #-----------------------------------------------------------------
    def control_callback_sg1(self):
        if not self.unity_ready:
            return

        if self.traj_finita:
            return
        
        if not self.traiettoria_pronta:
            return

        self.get_logger().info( f"[DEBUG] condizione_fine = {self.t >= self.traj_duration}, " f"t = {self.t:.3f}, traj_duration = {self.traj_duration:.3f}, traj_finita = {self.traj_finita}" )
        # Se la traiettoria è finita
        if self.traiettoria_pronta and self.traj_index >= len(self.full_traj) and not self.traj_finita:

            self.get_logger().info("[DEBUG] Entrato nel blocco fine traiettoria")
            self.traj_finita = True # <-- evita ripetizioni
            # Statistiche dt_real
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

            

            self.salva_dati(IAE, ISE, ITAE, RMSE, dt_mean, dt_max, dt_min,conteggio)
            self.get_logger().info(f"dt_real — media: {dt_mean:.6f}, max: {dt_max:.6f}, min: {dt_min:.6f}")

            return

        
        # Se Unity non è ancora connesso, non calcolare nulla
        if not self.traiettoria_pronta or not self.unity_connected or self.current_js is None:
            self.get_logger().info("In attesa di feedback da Unity...", throttle_duration_sec=2.0)
            return



        Xd, Xd_dot, Xd_ddot, Qd = self.evaluate()

        u, qd, qd_dot, qd_ddot = self.controller.compute_command(
            self.current_js,
            Xd, Xd_dot, Xd_ddot, Qd
        )

        # === 3. Stato attuale ===
        q = np.array(self.current_js.position)
        qdot = np.array(self.current_js.velocity) if len(self.current_js.velocity) == 6 else np.zeros(6)

        # === 4. Integrazione dinamica con dt_real ===
        qdot_new = qdot + u * self.dt
        q_new = q + qdot_new * self.dt

        # === 5. Pubblica verso Unity ===
        msg = Float64MultiArray()
        msg.data = q_new.tolist()
        self.publisher.publish(msg)
        self.t_traj += self.dt
        self.traj_index += 1

    def evaluate(self):
        i = int(self.t_traj/self.dt)
        if i >= len(self.full_traj):
            i = len(self.full_traj) - 1

        traj_point = self.full_traj[i]
        return traj_point.X, traj_point.Xdot, traj_point.Xddot, traj_point.Q

    #-----------------------------------------------------------------
    #   CONTROLLO SPAZIO OPERATIVO (CARTESIANO) CON FEEDBACK GIUNTI
    #-----------------------------------------------------------------
    def control_callback_so1(self):

        # === 1. Condizioni di sicurezza ===
        if not self.unity_ready:
            return

        if self.traj_finita:
            return

        if not self.traiettoria_pronta:
            return

        if not self.unity_connected or self.current_js is None:
            self.get_logger().info("In attesa di feedback da Unity...", throttle_duration_sec=2.0)
            return

        # === 2. Fine traiettoria ===
        if self.traj_index >= len(self.full_traj) and not self.traj_finita:

            self.get_logger().info("[DEBUG] Fine traiettoria raggiunta (SO).")
            self.traj_finita = True

            # Statistiche dt (qui usi dt fisso)
            dt_mean = self.dt
            dt_max = self.dt
            dt_min = self.dt

            # Indici di performance
            IAE, ISE, ITAE, RMSE = self.controller.compute_performance_indices(dt_mean)

            self.get_logger().info("=== INDICI DI PRESTAZIONE (SO) ===")
            self.get_logger().info(f"IAE  = {IAE:.6f}")
            self.get_logger().info(f"ISE  = {ISE:.6f}")
            self.get_logger().info(f"ITAE = {ITAE:.6f}")
            self.get_logger().info(f"RMSE = {RMSE:.6f}")
            self.get_logger().info("==================================")

            self.salva_dati(IAE, ISE, ITAE, RMSE, dt_mean, dt_max, dt_min)
            return

        # === 3. Punto di traiettoria corrente ===
        traj_point = self.full_traj[self.traj_index]

        # === 4. Comando cartesiano → velocità giunti ===
        qdot_cmd = self.controller.compute_command_so(self.current_js, traj_point)
        qdot_cmd = np.array(qdot_cmd, dtype=float)

        # Saturazione di sicurezza
        max_qdot = 1.5
        qdot_cmd = np.clip(qdot_cmd, -max_qdot, max_qdot)

        # === 5. Integrazione per ottenere q_cmd ===
        self.q_cmd = self.q_cmd + qdot_cmd * self.dt

        # === 6. Controllo sanità ===
        if np.any(~np.isfinite(self.q_cmd)):
            self.get_logger().error("q_cmd contiene NaN o Inf! Blocco il controllo.")
            return

        # Limiti giunti
        q_min = np.deg2rad([-180, -120, -120, -180, -120, -360])
        q_max = np.deg2rad([ 180,  120,  120,  180,  120,  360])
        self.q_cmd = np.clip(self.q_cmd, q_min, q_max)

        # === 7. Pubblica verso Unity ===
        msg = Float64MultiArray()
        msg.data = self.q_cmd.tolist()
        self.publisher.publish(msg)

        # === 8. Avanza nella traiettoria ===
        self.t_traj += self.dt
        self.traj_index += 1



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