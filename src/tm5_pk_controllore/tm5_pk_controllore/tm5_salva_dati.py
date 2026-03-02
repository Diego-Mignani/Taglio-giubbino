import numpy as np
import os
import csv
from datetime import datetime

class SalvaDati:
    def __init__(self):
        self.log_data = []

    def salva(self, dt_log, controller, get_logger, Kp, Kd, K_ori, w_ori):
            get_logger.info("[DEBUG] Fine traiettoria raggiunta (tempo)")
            dt_array = np.array(dt_log, dtype=float)
            dt_mean = np.mean(dt_log)
            dt_max = np.max(dt_log)
            dt_min = np.min(dt_log)
            conteggio = np.sum(dt_array > 0.12)

            IAE, ISE, ITAE, RMSE = controller.compute_performance_indices(dt_mean)
            get_logger.info("=== INDICI DI PRESTAZIONE ===")
            get_logger.info(f"IAE  = {IAE:.6f}")
            get_logger.info(f"ISE  = {ISE:.6f}")
            get_logger.info(f"ITAE = {ITAE:.6f}")
            get_logger.info(f"RMSE = {RMSE:.6f}")
            get_logger.info("==============================")

            get_logger.info(
                f"dt_real — media: {dt_mean:.6f}, max: {dt_max:.6f}, min: {dt_min:.6f}"
            )
            self.salva_prestazioni(IAE, ISE, ITAE, RMSE, dt_mean, dt_max, dt_min, conteggio, Kp, Kd, K_ori, w_ori)
            self.salva_log_traiettoria()


    def aggiorna_log_traiettoria(self, t_real, Xd, Xd_dot, Xd_ddot, X, Xdot_real, Qd, Q_act, omega_real):
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
        
    def salva_prestazioni(self, IAE, ISE, ITAE, RMSE, dt_mean, dt_max, dt_min, conteggio, Kp, Kd, K_ori, w_ori):
        save_dir = os.path.expanduser("~/ros2_ws/prestazioni")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "prestazioni_robot.csv")

        write_header = not os.path.exists(file_path)

        Kp_str = ",".join(map(str, np.diag(Kp).tolist()))
        Kd_str = ",".join(map(str, np.diag(Kd).tolist()))

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
                K_ori,
                w_ori,
                len(self.full_traj),
                dt_mean,
                dt_max,
                dt_min,
                conteggio
            ])

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


            