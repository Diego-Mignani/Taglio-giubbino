import numpy as np
from my_robot_utils import urdf as kdl_parser
from PyKDL import ChainFkSolverPos_recursive, ChainJntToJacSolver, Jacobian, JntArray, Frame


class KDLKinematics6DOF:
    def __init__(self, robot_desc: str,
                 base_link: str = "base",
                 ee_link: str = "flange"):
        """
        Inizializza la catena KDL a partire dall'URDF.

        :param robot_desc: stringa URDF (contenuto di robot_description)
        :param base_link: nome del link di base nel modello URDF
        :param ee_link: nome del link dell'end-effector (flange) nel modello URDF
        """
        ok, tree = kdl_parser.treeFromString(robot_desc)
        if not ok:
            raise RuntimeError("Impossibile caricare l'albero KDL dall'URDF")

        self.chain = tree.getChain(base_link, ee_link)
        self.num_joints = self.chain.getNrOfJoints()

        self.fk_solver = ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = ChainJntToJacSolver(self.chain)
        self._jac_kdl = Jacobian(self.num_joints)

    # ============================
    #       FORWARD KINEMATICS
    # ============================
    def fk_6dof(self, q):
        """
        Restituisce la matrice omogenea 4x4 dell'end-effector per le posizioni dei giunti date.

        :param q: array-like di lunghezza num_joints con le posizioni dei giunti
        :return: matrice 4x4 (numpy.ndarray)
        """
        q = np.asarray(q, dtype=float)
        if q.shape[0] != self.num_joints:
            raise ValueError(f"Attesi {self.num_joints} giunti, ricevuti {q.shape[0]}")

        joints = JntArray(self.num_joints)
        for i in range(self.num_joints):
            joints[i] = q[i]

        frame = Frame()
        self.fk_solver.JntToCart(joints, frame)

        T = np.eye(4)
        for i in range(3):
            for j in range(3):
                T[i, j] = frame.M[i, j]
            T[i, 3] = frame.p[i]

        return T

    # ============================
    #         JACOBIANA
    # ============================
    def get_full_jacobian(self, q):
        """
        Restituisce la Jacobiana 6xN completa (3 lin, 3 ang).

        :param q: array-like di lunghezza num_joints con le posizioni dei giunti
        :return: matrice 6 x num_joints (numpy.ndarray)
        """
        q = np.asarray(q, dtype=float)
        if q.shape[0] != self.num_joints:
            raise ValueError(f"Attesi {self.num_joints} giunti, ricevuti {q.shape[0]}")

        joints = JntArray(self.num_joints)
        for i in range(self.num_joints):
            joints[i] = q[i]

        self.jac_solver.JntToJac(joints, self._jac_kdl)

        J = np.zeros((6, self.num_joints))
        for i in range(6):
            for j in range(self.num_joints):
                J[i, j] = self._jac_kdl[i, j]

        return J

    # ============================
    #      INVERSE KINEMATICS
    # ============================
    def ik_position(self,
                    Xd: np.ndarray,
                    q_seed: np.ndarray,
                    max_iter: int = 100,
                    alpha: float = 0.5,
                    tol: float = 1e-4):
        """
        Risolve X(q) ~= Xd (solo posizione) con metodo iterativo (Jacobian pseudoinverse).

        :param Xd: posizione target (3,) in metri [x, y, z]
        :param q_seed: configurazione iniziale dei giunti (num_joints,)
        :param max_iter: massimo numero di iterazioni
        :param alpha: passo di aggiornamento (gain sul dq)
        :param tol: soglia sulla norma dell'errore di posizione
        :return: q soluzione (num_joints,)
        """
        Xd = np.asarray(Xd, dtype=float).reshape(3)
        q = np.asarray(q_seed, dtype=float).copy()
        if q.shape[0] != self.num_joints:
            raise ValueError(f"Attesi {self.num_joints} giunti, ricevuti {q.shape[0]}")

        for _ in range(max_iter):
            T = self.fk_6dof(q)
            X = self.position_from_T(T)
            e = Xd - X

            if np.linalg.norm(e) < tol:
                break

            J_full = self.get_full_jacobian(q)
            J_pos = J_full[:3, :]      # solo parte lineare

            # SVD per damped least squares
            U, S, Vt = np.linalg.svd(J_pos, full_matrices=False)

            lambda2 = 1e-4  # damping, puoi tararlo (1e-4–1e-2)
            S_damped = S / (S**2 + lambda2)

            J_pos_pinv = (Vt.T * S_damped) @ U.T

            dq = alpha * (J_pos_pinv @ e)

            # Limite sul passo per evitare salti grossi in q
            max_delta_q = np.deg2rad(5.0)  # max 5° per iterazione
            dq = np.clip(dq, -max_delta_q, max_delta_q)

            q += dq


        return q
    
    def position_from_T(self, T):
        """
        Estrae la posizione (x, y, z) dalla matrice omogenea 4x4.

        :param T: matrice omogenea 4x4
        :return: array (3,) con [x, y, z]
        """
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError("La matrice T deve essere di forma (4, 4)")
        return T[:3, 3]
