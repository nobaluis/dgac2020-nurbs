import matplotlib.pyplot as plt
import numpy as np

import bspline
import examples_data

# load NURBS data example
data = examples_data.nurbs()[1]

# init the plot
plt.figure(figsize=(6.4, 4.8), dpi=150)

# create the splines
for i in range(5):
    p = data.degree
    p_list = data.points
    u_list = data.knots
    w_list = data.weights
    w_list[2] = i  # edit the middle weight
    u = np.linspace(u_list[0], u_list[-1], 100)
    spline = bspline.Nurbs(p, p_list, u_list, w_list, u)

    # plot the spline
    plt.plot(spline.curve[:, 0], spline.curve[:, 1])

# plot the control points
plt.scatter(p_list[:, 0], p_list[:, 1], color='red')
for i in range(len(p_list)):
    plt.annotate(r'$P_%d$' % i, p_list[i, :], textcoords="offset points", xytext=(0, 5), ha='center')

# add labels
plt.legend([r'$C(u),  w_2=%d$' % (i,) for i in range(5)], bbox_to_anchor=(1, 1.016))

plt.tight_layout()  # layout adjust
# plt.savefig('../images/weights_effect.png')  # save the fig
plt.show()  # show the fig
