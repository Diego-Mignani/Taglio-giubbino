import numpy as np
from geometry_msgs.msg import Pose

def generate_zip(start, end, num_points=100, amplitude=0.02, frequency=3.0):
    t = np.linspace(0, 1, num_points)
    line = start + np.outer(t, (end - start))

    direction = end[:2] - start[:2]
    direction = direction / np.linalg.norm(direction)

    normal = np.array([-direction[1], direction[0]])
    wave = amplitude * np.sin(2 * np.pi * frequency * t)

    line[:, 0] += wave * normal[0]
    line[:, 1] += wave * normal[1]

    poses = []
    for p in line:
        pose = Pose()
        pose.position.x = float(p[0])
        pose.position.y = float(p[1])
        pose.position.z = float(p[2])
        pose.orientation.w = 1.0
        poses.append(pose)

    return poses
