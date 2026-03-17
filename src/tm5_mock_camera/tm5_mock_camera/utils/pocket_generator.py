import numpy as np
from geometry_msgs.msg import Pose

def generate_pocket(width=0.12, height=0.18, z=0.15, n=25, noise=0.002, offset=(0.5, 0.3)):
    pts = []

    # lato sinistro
    for i in range(n):
        y = i * (height / n)
        pts.append([0, y, z])

    # arco superiore
    t = np.linspace(0, np.pi, n)
    for ti in t:
        x = (width/2) * (1 - np.cos(ti))
        y = height
        pts.append([x, y, z])

    # lato destro
    for i in range(n):
        y = height - i * (height / n)
        pts.append([width, y, z])

    # arco inferiore
    t = np.linspace(np.pi, 2*np.pi, n)
    for ti in t:
        x = (width/2) * (1 - np.cos(ti))
        y = 0
        pts.append([x, y, z])

    pts = np.array(pts)
    pts[:, 0] += noise * np.random.randn(len(pts))
    pts[:, 1] += noise * np.random.randn(len(pts))
    pts[:, 0] += offset[0]
    pts[:, 1] += offset[1]

    poses = []
    for p in pts:
        pose = Pose()
        pose.position.x = float(p[0])
        pose.position.y = float(p[1])
        pose.position.z = float(p[2])
        pose.orientation.w = 1.0
        poses.append(pose)

    return poses
