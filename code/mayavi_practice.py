import numpy as np
from mayavi import mlab

import examples_data
from bspline import Nurbs


def plot_example(curve, points):
    colors = [(.2, .521, 1), (1, .356, .2)]
    mlab.figure(size=(640, 480))
    if points.shape[1] == 2:
        mlab.plot3d(curve[:, 0], curve[:, 1], np.zeros_like(curve[:, 0]), tube_radius=0.025, color=colors[0])
        mlab.points3d(points[:, 0], points[:, 1], np.zeros_like(points[:, 0]), scale_factor=0.1, color=colors[1])
    elif points.shape[1] == 3:
        mlab.plot3d(curve[:, 0], curve[:, 1], curve[:, 2], tube_radius=0.025, color=colors[0])
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.1, color=colors[1])
    mlab.show()


if __name__ == '__main__':
    # Load nurbs examples
    nurbs_ex = examples_data.nurbs()

    # Example 01 - circle
    p = nurbs_ex[0].degree
    p_list = nurbs_ex[0].points
    u_list = nurbs_ex[0].knots
    w_list = nurbs_ex[0].weights
    u = np.linspace(u_list[0], u_list[-1], 100)
    spline = Nurbs(p, p_list, u_list, w_list, u)
    plot_example(spline.curve, p_list)
