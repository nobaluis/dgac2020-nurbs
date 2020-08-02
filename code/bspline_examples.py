import bspline
import examples_data
import numpy as np
import matplotlib.pyplot as plt


def plot_example(p, p_list, c_u, nip_u, u, ex_number, save_figs=False):
    # plot the curve & control points}
    plt.figure(figsize=(6.4, 4.8), dpi=150)
    plt.plot(c_u[:, 0], c_u[:, 1])
    plt.plot(p_list[:, 0], p_list[:, 1], 'k--', linewidth=0.9, alpha=0.4)
    plt.scatter(p_list[:, 0], p_list[:, 1], color='red')
    # plt.legend([r'$C(u)$'], loc='lower right')
    for i in range(len(p_list)):
        plt.annotate(r'$P_%d$' % i, p_list[i, :],
                     textcoords="offset points", xytext=(0, 5), ha='center')
    plt.axis('equal')
    plt.tight_layout()
    if save_figs:
        plt.savefig('../images/ex_%02d_curve.png' % ex_number)
    plt.show()

    # plot the basis functions
    plt.figure(figsize=(6.4, 4.8), dpi=150)
    plt.plot(u, nip_u.T)
    plt.xlabel(r'$u$')
    if len(p_list) <= 10:
        plt.legend([r'$N_{%d,%d}(u)$' % (i, p) for i in range(len(p_list))],
                   bbox_to_anchor=(1, 1.016))
    plt.tight_layout()
    if save_figs:
        plt.savefig('../images/ex_%02d_basis_funcs.png' % ex_number)
    plt.show()


if __name__ == '__main__':
    # B-spline examples
    bsplines = examples_data.bspline()
    for i in range(len(bsplines)):
        p = bsplines[i].degree
        p_list = bsplines[i].points
        u_list = bsplines[i].knots
        u = np.linspace(u_list[0], u_list[-1], 100)
        spline = bspline.Bspline(p, p_list, u_list)
        points = spline.points(u)
        basis_funcs = bspline.get_basis_vector(u, u_list, p)
        plot_example(p, p_list, points, basis_funcs, u, i, save_figs=False)

    # NURBS examples
    nurbs = examples_data.nurbs()
    for i in range(len(nurbs)):
        p = nurbs[i].degree
        p_list = nurbs[i].points
        u_list = nurbs[i].knots
        w_list = nurbs[i].weights
        u = np.linspace(u_list[0], u_list[-1], 100)
        spline = bspline.Nurbs(p, p_list, u_list, w_list)
        points = spline.points(u)
        basis_funcs = bspline.get_basis_vector(u, u_list, p)
        plot_example(p, p_list, points, basis_funcs, u, i + len(bsplines), save_figs=False)
