# perception/mock_source.py
import numpy as np

class PerceptionSource:
    """
    Per ora simula la rete neurale producendo punti 3D grezzi.
    In futuro leggerai i punti reali dal sistema di visione.
    """
    def get_raw_points(self):
        # Esempio: linea retta nello spazio
        n_points = 10
        x = np.linspace(0.3, 0.5, n_points)
        y = np.linspace(0.0, 0.2, n_points)
        z = 0.1 * np.ones(n_points)
        points = np.vstack((x, y, z)).T  # shape (n_points, 3)
        return points
