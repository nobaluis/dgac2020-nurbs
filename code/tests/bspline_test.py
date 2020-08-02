import unittest

import numpy as np

import occ_utils
from bspline import Bspline, BsplineSurface, Nurbs, NurbsSurface


def pseudo_random_surf():
    p, q = 2, 2
    u_list = np.array([0, 0, 0, 0.25, 0.75, 1, 1, 1])
    v_list = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])
    p_x = np.linspace(0.2, 0.85, len(u_list) - p - 1)
    p_y = np.linspace(-0.5, 0.5, len(v_list) - q - 1)
    p_z = np.random.random(size=(len(p_x), len(p_y),))
    p_matrix = np.empty((len(p_x), len(p_y), 3))
    for i in range(len(p_x)):
        for j in reversed(range(len(p_y))):
            p_matrix[i, j, :] = [p_x[i], p_y[j], (p_z[i, j] * 0.2) + 0.7]
    return p, q, p_matrix, u_list, v_list


class BsplineTest(unittest.TestCase):
    def test_bspline_curve(self):
        # 1. Prepare the data
        p = 3
        p_list = np.array([[0, 1], [2, 4], [4, 1], [5, 2], [7, 3], [10, 8], [12, 7], [14, 0]], dtype='d')
        u_list = np.array([0, 0, 0, 0, 1, 2, 3, 3, 4, 4, 4, 4], dtype='d')
        u = np.linspace(u_list[0], u_list[-1], 100)
        # 2. Compute reference values
        curve_ref = occ_utils.bspline(p_list, u_list)
        points_ref = occ_utils.curve_dn(curve_ref, u, 0)  # zero order derivative = curve points
        derivatives_ref = occ_utils.curve_dn(curve_ref, u, 1)
        points_ref = points_ref[:, 0:2]
        derivatives_ref = derivatives_ref[:, 0:2]
        # 3. Compute test values
        curve = Bspline(p, p_list, u_list)
        points = curve.points(u)
        derivatives = curve.derivatives(u, 1)
        # 4. Do the comparative
        points_test = np.isclose(points_ref, points, atol=1e-10)
        derivatives_test = np.isclose(derivatives_ref, derivatives, atol=1e-10)
        self.assertTrue(np.all(points_test) and np.all(derivatives_test))

    def test_bspline_surface(self):
        # 1. Prepare the data
        p, q, p_matrix, u_list, v_list = pseudo_random_surf()
        u = np.linspace(u_list[0], u_list[-1], 50)
        v = np.linspace(v_list[0], v_list[-1], 60)
        # 2. Compute reference values
        surf_ref = occ_utils.bspline_surf(p_matrix, u_list, v_list)
        points_ref = occ_utils.surf_dn(surf_ref, u, v, 0, 0)  # derivatives zero order = points on surface
        du_ref = occ_utils.surf_dn(surf_ref, u, v, 1, 0)  # partial derivatives over u direction
        dv_ref = occ_utils.surf_dn(surf_ref, u, v, 0, 1)  # partial derivatives over v direction
        # 3. Compute test values
        surf = BsplineSurface(p, q, p_matrix, u_list, v_list)
        points = surf.points(u, v)
        du = surf.derivatives(u, v, 1, 0)
        dv = surf.derivatives(u, v, 0, 1)
        # 4. Do the comparative
        points_test = np.isclose(points_ref, points, atol=1e-10)
        derivatives_u_test = np.isclose(du_ref, du, atol=1e-10)
        derivatives_v_test = np.isclose(dv_ref, dv, atol=1e-10)
        self.assertTrue(np.all(points_test) and np.all(derivatives_u_test) and np.all(derivatives_v_test))

    def test_nurbs_curve(self):
        # 1. Prepare the data
        p = 2
        p_list = np.array([[1, 0], [1, 1], [-1, 1], [-1, 0], [-1, -1], [1, -1], [1, 0]], dtype='d')
        w_list = np.array([1., .5, .5, 1., .5, .5, 1.])
        u_list = np.array([0., 0., 0., .25, .5, .5, .75, 1., 1., 1.])
        u = np.linspace(u_list[0], u_list[-1], 100)
        # 2. Compute reference values
        curve_ref = occ_utils.nurbs(p_list, u_list, w_list)
        points_ref = occ_utils.curve_dn(curve_ref, u, 0)
        derivatives_ref = occ_utils.curve_dn(curve_ref, u, 1)
        points_ref = points_ref[:, 0:2]  # drop z dimension
        derivatives_ref = points_ref[:, 0:2]
        # 3. Compute test values
        curve = Nurbs(p, p_list, u_list, w_list)
        points = curve.points(u)
        # 4. Do the comparative
        points_test = np.isclose(points_ref, points, atol=1e-10)
        # TODO: test for derivatives
        self.assertTrue(np.all(points_test))

    def test_nurbs_surface(self):
        # 1. Prepare the data
        p, q, p_matrix, u_list, v_list = pseudo_random_surf()
        w_matrix = np.random.random(size=(p_matrix.shape[0], p_matrix.shape[1]))
        u = np.linspace(u_list[0], u_list[-1], 50)
        v = np.linspace(v_list[0], v_list[-1], 60)
        # 2. Compute reference values
        surf_ref = occ_utils.nurbs_surf(p_matrix, w_matrix, u_list, v_list)
        points_ref = occ_utils.surf_dn(surf_ref, u, v, 0, 0)  # zero order derivatives = surface points
        # 3. Compute test values
        surf = NurbsSurface(p, q, p_matrix, w_matrix, u_list, v_list)
        points = surf.points(u, v)
        # 4. Do the comparative
        points_test = np.isclose(points_ref, points, atol=1e-10)
        # TODO: test for derivatives
        self.assertTrue(np.all(points_test))
