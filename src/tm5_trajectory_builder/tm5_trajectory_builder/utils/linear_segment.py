import numpy as np
from geometry_msgs.msg import Pose

def linear_cartesian_segment(p_start: Pose, p_end: Pose, steps=50):
    """
    Genera una lista di Pose che interpolano linearmente tra p_start e p_end.
    """

    # Posizioni
    start = np.array([p_start.position.x, p_start.position.y, p_start.position.z])
    end   = np.array([p_end.position.x,   p_end.position.y,   p_end.position.z])

    # Orientazioni (per ora costanti)
    q_start = p_start.orientation
    q_end   = p_end.orientation

    poses = []

    for t in np.linspace(0, 1, steps):
        p = start + t * (end - start)

        pose = Pose()
        pose.position.x = float(p[0])
        pose.position.y = float(p[1])
        pose.position.z = float(p[2])

        # Manteniamo l'orientazione costante (si può migliorare con SLERP)
        pose.orientation = q_start

        poses.append(pose)

    return poses
    