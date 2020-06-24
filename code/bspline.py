import numpy as np


def div(n, d):
    """Cox deBoor special division"""
    return n / d if d else 0


def basis(i, p, u, u_list):
    """Cox deBoor formula

    Parameters
    ----------
    i: int
        The index of basis function related to ith control point
    p: int
        The degree of basis function
    u: double
        Parametric value to evaluate the function
    u_list: numpy.array
        The knot vector
    """
    if p == 0:
        return 1 if u_list[i] <= u < u_list[i + 1] else 0
    else:
        a = div((u - u_list[i]), (u_list[i + p] - u_list[i])) * basis(i, p - 1, u, u_list)
        b = div((u_list[i + p + 1] - u), (u_list[i + p + 1] - u_list[i + 1])) * basis(i + 1, p - 1, u, u_list)
        return a + b


basis_vector = np.vectorize(basis, excluded=['u_list'])  # vectorized version of basis fucntion


def point_eval(p, p_list, u_list, u):
    """Evaluate the points in the curve over the parametric variable and knot vector

    Parameters
    ----------
    p: int
        Degree of curve
    p_list: numpy.array
        Set of control points
    u: numpy.array
        The parametric variable
    u_list: numpy.array
        The knot vector, dimension must be m = n + p + 1

    Return
    ------
    curve: numpy.array
        The points in the curve over u
    basis_functions: numpy.array
        The B-spline basis functions over u
    """
    # Check dimensions
    n, d = p_list.shape
    m, length = len(u_list), len(u)
    if m != (n + p + 1):
        raise ValueError('Wrong knots dimension')

    # Evaluate points
    u[-1] -= 1e-10  # for stability
    basis_functions = np.empty((n, length))
    curve = np.zeros((length, d))
    for i in range(n):
        basis_ip = basis_vector(i, p, u, u_list=u_list)
        basis_functions[i, :] = basis_ip
        curve += p_list[i] * basis_ip[:, None]
    return curve, basis_functions


def do_perspective_map(p_list, w_list):
    pw_list = p_list * w_list[:, None]
    pw_list = np.column_stack((pw_list, w_list))
    return pw_list


def undo_perspective_map(pw_list):
    return pw_list[:, :-1] / pw_list[:, -1, None]


class Bspline(object):
    """Computes generic B-Spline

    Parameters
    ----------
    p: int
        Degree of curve
    p_list: numpy.array
        Set of control points
    u_list: numpy.array
        Knots vector, dimension must be m=n+p+1
    u: numpy.array
        Parametric coordinates vector
    """

    def __init__(self, p, p_list, u_list, u):
        self.p = p
        self.p_list = p_list
        self.u_list = u_list
        self.curve, self.basis_funcs = point_eval(p, p_list, u_list, u)


class Nurbs(Bspline):
    """Computes NURBS curve using perspective map

    Parameters
    ----------
    p: int
        Degree of curve
    p_list: numpy.array
        Set of control points
    u_list: numpy.array
        Knots vector, dimension must be m=n+p+1
    w_list: numpy.array
        Wights vector
    u: numpy.array
        Parametric coordinates vector
    """

    def __init__(self, p, p_list, u_list, w_list, u):
        self.w_list = w_list
        pw_list = do_perspective_map(p_list, w_list)
        super(Nurbs, self).__init__(p, pw_list, u_list, u)
        self.curve = undo_perspective_map(self.curve)
