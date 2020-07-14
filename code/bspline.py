import numpy as np

EPSILON = 1e-10


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


basis_vector = np.vectorize(basis, excluded=['u_list'])  # vectorized version of basis function


def basis_d(i, p, u, u_list):
    """Computes the derivative of the basis function of degree p

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
    a = div(p - 1, u_list[i + p] - u_list[i]) * basis(i, p - 1, u, u_list)
    b = div(p - 1, u_list[i + p + 1] - u_list[i + 1]) * basis(i + 1, p - 1, u, u_list)
    return a - b


basis_d_vector = np.vectorize(basis_d, excluded=['u_list'])  # vectorized version of basis_d function


def point_eval(p, p_list, u_list, u):
    """Evaluate the points in the B-Spline of degree p over the parametric variable u

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
        raise ValueError('Wrong knot vector dimension')

    # Evaluate points
    u[-1] -= 1e-10  # for stability
    basis_functions = np.empty((n, length))
    curve = np.zeros((length, d))
    for i in range(n):
        basis_ip = basis_vector(i, p, u, u_list=u_list)
        basis_functions[i, :] = basis_ip
        curve += p_list[i] * basis_ip[:, None]
    return curve, basis_functions


def derivatives_eval(p, p_list, u_list, u):
    """Evaluate the first derivative of the B-Spline of degree p

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
    curve_derivatives: numpy.array
        The first derivative of the curve over u
    basis_derivatives: numpy.array
        The derivative of B-spline basis functions over u
    """
    n, d = p_list.shape
    m, length = len(u_list), len(u)

    u[-1] -= EPSILON  # for stability
    basis_derivatives = np.empty((n, length))
    curve_derivatives = np.zeros((length, d))
    for i in range(n):
        basis_d_ip = basis_d_vector(i, p, u, u_list=u_list)
        basis_derivatives[i, :] = basis_d_ip
        curve_derivatives += p_list * basis_d_ip[:, None]
    return curve_derivatives, basis_derivatives


def surface_eval(p, q, p_matrix, u_list, v_list, u, v):
    """Computes the B-Spline surface over u and v parametric variables

    p, q: int
        Degrees of functions
    p_matrix: numpy.array
        The matrix of control points
    u_list, v_list: numpy.array
        The knots vectors of u and v
    u, v: numpy.array
        The parametric coordinates vectors

    Return
    ------
    surface: numpy.array
        The grid of points over u and v
    basis_u, basis_v: numpy.array
        The basis functions of degree p and q respectively
    """
    n, m, d = p_matrix.shape
    r, s = len(u_list), len(v_list)
    if r != (n + p + 1) or s != (m + q + 1):
        raise ValueError('Wrong knot vector(s) dimension')

    # Evaluate points on surface
    u[-1] -= EPSILON
    v[-1] -= EPSILON
    basis_u = np.empty((n, len(u)))
    basis_du = np.empty((n, len(u)))
    basis_v = np.empty((m, len(v)))
    basis_dv = np.empty((m, len(v)))

    # Compute basis functions
    for i in range(n):
        basis_u[i, :] = basis_vector(i, p, u, u_list=u_list)
        basis_du[i, :] = basis_d_vector(i, p, u, u_list=u_list)
    for j in range(m):
        basis_v[j, :] = basis_vector(j, q, v, u_list=v_list)
        basis_dv[j, :] = basis_d_vector(j, q, v, u_list=v_list)

    # Compute the tensor product
    surface = np.einsum('ni,nmd,mj->ijd', basis_u, p_matrix, basis_v)

    return surface, basis_u, basis_v, basis_du, basis_dv


def homogeneous_coord(p_list, w_list):
    """Express the the control points in homogeneous coordinates

    Parameters
    ----------
    p_list: numpy.array
        The set of control points
    w_list: numpy.array
        The weights vector
    """
    pw_list = p_list * w_list[:, None]
    pw_list = np.column_stack((pw_list, w_list))
    return pw_list


def perspective_map(pw_list):
    """Do the perspective map (H)

    Parameters
    ----------
    pw_list: numpy.array
        Set of control points expressed in high order dimension
    """
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
        self.u = u
        self.curve, self.basis_funcs = point_eval(p, p_list, u_list, u)

    def derivatives(self):
        """Computes the first derivatives"""
        return derivatives_eval(self.p, self.p_list, self.u_list, self.u)


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
        pw_list = homogeneous_coord(p_list, w_list)
        super(Nurbs, self).__init__(p, pw_list, u_list, u)
        self.curve = perspective_map(self.curve)

    def derivatives(self):
        raise NotImplementedError('This function is not ready')


class BsplineSurface(object):
    def __init__(self, p, q, p_matrix, u_list, v_list, u, v):
        self.p = p
        self.q = q
        self.p_matrix = p_matrix
        self.u_list = u_list
        self.v_list = v_list
        self.u = u
        self.v = v
        self.surface, self.basis_u, self.basis_v, self.basis_du, self.basis_dv = \
            surface_eval(p, q, p_matrix, u_list, v_list, u, v)

    def normal(self):
        """Computes the surface normal of the B-Spline surface"""
        s_du = np.einsum('ni,nmd,mj->ijd', self.basis_du, self.p_matrix, self.basis_v)
        s_dv = np.einsum('ni,nmd,mj->ijd', self.basis_u, self.p_matrix, self.basis_dv)
        return np.cross(s_du, s_dv)


class NurbsSurface(BsplineSurface):
    def __init__(self, p, q, p_matrix, w_matrix, u_list, v_list, u, v):
        self.w_matrix = w_matrix
        # reshape set of points and express in h. coord.
        p_shape = p_matrix.shape
        pw_list = homogeneous_coord(p_matrix.reshape(p_shape[0] * p_shape[1], p_shape[2]), w_matrix.reshape(-1))
        # build the b-spline
        super().__init__(p, q, pw_list.reshape(p_shape), u_list, v_list, u, v)
        # perspective map
        self.p_matrix = p_matrix
        surf_shape = self.surface.shape
        self.surface = perspective_map(self.surface.reshape(surf_shape[0] * surf_shape[1], surf_shape[2]))
        self.surface.shape = surf_shape
