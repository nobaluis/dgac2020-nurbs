# %% Creates the data example
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bspline import BsplineSurface

p = 2
q = 2
u_list = np.array([0, 0, 0, 0.25, 0.75, 1, 1, 1])
v_list = np.array([0, 0, 0, 0.25, 0.25, 0.5, 1, 1, 1])
p_x = np.linspace(0, 1, len(u_list) - p - 1)
p_y = np.linspace(0, 1, len(v_list) - q - 1)
p_z = np.random.random(size=(len(p_x), len(p_y),))
p_matrix = np.empty((len(p_x), len(p_y), 3))
for i in range(len(p_x)):
    for j in reversed(range(len(p_y))):
        p_matrix[i, j, :] = [p_x[i], p_y[j], p_z[i, j]]
# p_matrix = np.random.normal(size=(len(u_list) - p - 1, len(v_list) - q - 1, 3))
u = np.linspace(u_list[0], u_list[-1], 50)
v = np.linspace(v_list[0], v_list[-1], 50)

surface = BsplineSurface(p, q, p_matrix, u_list, v_list)
points = surface.points(u, v)
normal = surface.normal(u, v)

# %% Naive plot
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# for i in range(len(u)):
#     for j in range(len(v)):
#         x = points[i, j, 0]
#         y = points[i, j, 1]
#         z = points[i, j, 2]
#         ax.scatter3D(x, y, z, color='blue')
# fig.show()

# %% Efficient plot
matplotlib.use("TkAgg")
n, m, d = points.shape
points.shape = (n * m, d)
x, y, z = points[:, 0], points[:, 1], points[:, 2]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z, c=z, cmap='viridis')
fig.show()

# %% Normal field plot
normal.shape = (n * m, d)
n0, n1, n2 = normal[:, 0], normal[:, 1], normal[:, 2]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.quiver(x, y, z, n0, n1, n2, length=0.09, color='red')
fig.show()

# %% OCC Surface
import occ_utils

bspline_surf = occ_utils.bspline_surf(p_matrix, u_list, v_list)

# %% OCC Display
from OCC.Display.SimpleGui import init_display

display, start_display, _, _ = init_display()
display.DisplayShape(bspline_surf, update=True)
start_display()
