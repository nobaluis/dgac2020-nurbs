import numpy as np

EPSILON = 1e-15


def div(n, d):
    """Cox deBoor special division"""
    return n / d if d else 0.


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
        return 1. if u_list[i] <= u < u_list[i + 1] else 0.
    else:
        a = div((u - u_list[i]), (u_list[i + p] - u_list[i])) * basis(i, p - 1, u, u_list)
        b = div((u_list[i + p + 1] - u), (u_list[i + p + 1] - u_list[i + 1])) * basis(i + 1, p - 1, u, u_list)
        return a + b


basis_vector = np.vectorize(basis, excluded=['u_list'])  # vectorized version of basis function


def basis_dk(k, i, p, u, u_list):
    """Computes the kth derivative of basis function fo degree p"""
    if k > p:
        raise ValueError('k should not exceed p')
    elif k == 0:
        return basis(i, p, u, u_list)
    a = div(p, u_list[i + p] - u_list[i]) * basis_dk(k - 1, i, p - 1, u, u_list)
    b = div(p, u_list[i + p + 1] - u_list[i + 1]) * basis_dk(k - 1, i + 1, p - 1, u, u_list)
    return a - b


basis_dk_vector = np.vectorize(basis_dk, excluded=['u_list'])  # vectorized version of basis_dk function


def get_basis_vector(u, u_list, p):
    """Computes the basis functions of degree p over all values of u"""
    u[-1] -= EPSILON  # for numeric stability
    n = len(u_list) - p - 1
    basis_v = np.empty((n, len(u)))
    for i in range(n):
        basis_v[i, :] = basis_vector(i, p, u, u_list=u_list)
    return basis_v


def get_basis_dk_vector(u, u_list, p, k):
    """Computes the kth derivative of basis function of degree p over all values of u"""
    u[-1] -= EPSILON  # for numeric stability
    n = len(u_list) - p - 1
    basis_dk_v = np.empty((n, len(u)))
    for i in range(n):
        basis_dk_v[i, :] = basis_dk_vector(k, i, p, u, u_list=u_list)
    return basis_dk_v


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
    """

    def __init__(self, p, p_list, u_list):
        # Check dimensions
        n, m = len(p_list), len(u_list)
        if m != (n + p + 1):
            raise ValueError('Wrong knot vector dimension')
        self.p = p
        self.p_list = p_list
        self.u_list = u_list

    def points(self, u):
        """Computes the points of B-Spline curve"""
        basis_v = get_basis_vector(u, self.u_list, self.p)
        curve_points = np.zeros((len(u), self.p_list.shape[1]))
        for i in range(len(self.p_list)):
            curve_points += self.p_list[i] * basis_v[i, :, None]
        return curve_points

    def derivatives(self, u, k):
        """Computes the kth derivative of B-Spline curve"""
        basis_d_v = get_basis_dk_vector(u, self.u_list, self.p, k)
        curve_derivatives = np.zeros((len(u), self.p_list.shape[1]))
        for i in range(len(self.p_list)):
            curve_derivatives += self.p_list[i] * basis_d_v[i, :, None]
        return curve_derivatives


class Nurbs(Bspline):
    """Computes NURBS curve

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
    """

    def __init__(self, p, p_list, u_list, w_list):
        self.w_list = w_list
        super(Nurbs, self).__init__(p, p_list, u_list)

    def points(self, u):
        p_list_backup = self.p_list.copy()  # original control points
        self.p_list = homogeneous_coord(self.p_list, self.w_list)  # c. points in homogeneous coord.
        curve_points = super().points(u)  # compute points with regular B-Spline
        self.p_list = p_list_backup  # restore original c. points
        return perspective_map(curve_points)  # perspective map to points

    def derivatives(self):
        raise NotImplementedError('This function is not ready')


class BsplineSurface(object):
    """B-Spline surface class

    Parameters
    ----------
    p: int
        Degree in u direction
    q: int
        Degree in v direction
    p_matrix: numpy.array
        Matrix of control points
    u_list: numpy.array
        Knots vector of u
    v_list: numpy.array
        Knots vector of v
    """

    def __init__(self, p, q, p_matrix, u_list, v_list):
        # Verify knot vectors dimensions
        n, m, d = p_matrix.shape
        r, s = len(u_list), len(v_list)
        if r != (n + p + 1) or s != (m + q + 1):
            raise ValueError('Wrong knot vector(s) dimension')
        self.p = p
        self.q = q
        self.p_matrix = p_matrix
        self.u_list = u_list
        self.v_list = v_list

    def points(self, u, v):
        """Computes te points in surface over u and v coordinates

        Parameters
        ----------
        u: numpy.array
            The coordinates of u
        v: numpy.array
            The coordinates of v
        """
        basis_u = get_basis_vector(u, self.u_list, self.p)
        basis_v = get_basis_vector(v, self.v_list, self.q)
        return np.einsum('ni,nmd,mj->ijd', basis_u, self.p_matrix, basis_v)

    def derivatives(self, u, v, k, l):
        """Computes the partial derivatives of any order d in [0, k+l]

        Parameters
        ----------
        u: numpy.array
            The coordinates of u
        v: numpy.array
            The coordinates of v
        k: int
            The order of derivative in u direction, k <= p
        l: int
            The order of derivative in v direction, l <= q
        """
        basis_k = get_basis_dk_vector(u, self.u_list, self.p, k)
        basis_l = get_basis_dk_vector(v, self.v_list, self.q, l)
        return np.einsum('ni,nmd,mj->ijd', basis_k, self.p_matrix, basis_l)

    def normal(self, u, v, unit=True):
        """Computes the surface normal of the B-Spline surface

        Parameters
        ----------
        u: numpy.array
            The coordinates of u
        v: numpy.array
            The coordinates of v
        unit: bool, optional
            If is true return the unit normal field
        """
        s_du = self.derivatives(u, v, 1, 0)
        s_dv = self.derivatives(u, v, 0, 1)
        normal = np.cross(s_du, s_dv)
        if unit:
            normal /= np.linalg.norm(normal, axis=2)[:, :, None]
        return normal


class NurbsSurface(BsplineSurface):
    def __init__(self, p, q, p_matrix, w_matrix, u_list, v_list):
        self.w_matrix = w_matrix
        super().__init__(p, q, p_matrix, u_list, v_list)

    def points(self, u, v):
        p_matrix_backup = self.p_matrix.copy()  # backup of original matrix
        m, n, _ = self.p_matrix.shape  # original c. points shape
        self.p_matrix = homogeneous_coord(self.p_matrix.reshape(m * n, 3), self.w_matrix.reshape(-1))  # h. coord.
        self.p_matrix.shape = (m, n, 4)  # reshape teh tensor
        surf_points = super().points(u, v)  # compute the b-spline surface in homogeneous coord.
        surf_points = perspective_map(surf_points.reshape(len(u) * len(v), 4))  # perspective map, 4D -> 3D
        surf_points.shape = (len(u), len(v), 3)  # reshape the tensor
        self.p_matrix = p_matrix_backup  # restore original c. points
        return surf_points

    def derivatives(self, u, v, k, l):
        raise NotImplementedError('This function is not ready')

    def normal(self, u, v):
        raise NotImplementedError('This function is not ready')