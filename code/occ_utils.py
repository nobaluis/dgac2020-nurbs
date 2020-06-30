import numpy as np
from OCC.Core.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger, TColStd_Array2OfInteger, \
    TColStd_Array2OfReal
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt


def np2occ_points(pts):
    """Converts numpy array of points into occ points array
    """
    if len(pts.shape) == 2:
        n, m = pts.shape
        occ_array = TColgp_Array1OfPnt(1, n)
        for i in range(n):
            if m == 2:
                occ_array.SetValue(i + 1, gp_Pnt(pts[i, 0], pts[i, 1], 0.))
            elif m == 3:
                occ_array.SetValue(i + 1, gp_Pnt(pts[i, 0], pts[i, 1], pts[i, 2]))
        return occ_array
    elif len(pts.shape) == 3:
        n, m, _ = pts.shape
        occ_array = TColgp_Array2OfPnt(1, n, 1, m)
        for i in range(n):
            for j in range(m):
                occ_array.SetValue(i + 1, j + 1, gp_Pnt(pts[i, j, 0], pts[i, j, 1], pts[i, j, 2]))
        return occ_array
    else:
        raise ValueError('Wrong points dimension')


def np2occ(data, container, cast):
    """Converts numpy array into a specific occ array

    Parameters
    ----------
    data: numpy.array
        The data to be converted
    container: OCC.Core.TColStd.*
        The target container of data
    cast: data type
        Type of data to force cast of data values
    """
    if len(data.shape) == 1:
        array = container(1, len(data))
        for i in range(len(data)):
            array.SetValue(i + 1, cast(data[i]))
    elif len(data.shape) == 2:
        array = container(1, len(data), 1, len(data[0]))
        for i in range(len(data)):
            for j in range(len(data[0])):
                array.SetValue(i + 1, j + 1, cast(data[i, j]))
    else:
        raise ValueError('Array data dimension not supported')
    return array


def np2occ_auto(v):
    """Converts automatic numpy array into a occ array in base of dtype"""
    data_type = v.dtype
    if data_type == np.dtype('float64'):
        if len(v.shape) == 1:
            return np2occ(v, TColStd_Array1OfReal, float)
        elif len(v.shape) == 2:
            return np2occ(v, TColStd_Array2OfReal, float)
        else:
            raise ValueError('Array data dimension not supported')
    elif data_type == np.dtype('int64'):
        if len(v.shape) == 1:
            return np2occ(v, TColStd_Array1OfInteger, int)
        elif len(v.shape) == 2:
            return np2occ(v, TColStd_Array2OfInteger, int)
        else:
            raise ValueError('Array data dimension not supported')
    else:
        raise ValueError('data type not supported')


def nurbs(p_list, u_list, w_list):
    """Construct NURBS curve from OCC with numpy arrays

    Parameters
    ----------
    p_list: numpy.array
        The set of control points
    u_list: numpy.array
        The knots vector
    w_list: numpy.array
        The weighs vector
    """
    u_vals, u_mults = np.unique(u_list, return_counts=True)
    p = len(u_list) - len(p_list) - 1
    poles = np2occ_points(p_list)
    weighs = np2occ_auto(w_list)
    knots = np2occ_auto(u_vals)
    mults = np2occ_auto(u_mults)
    return Geom_BSplineCurve(poles, weighs, knots, mults, p)


def bspline(p_list, u_list):
    """Builds B-spline curve from OCC with numpy arrays

    Parameters
    ----------
    p_list: numpy.array
        The set of control points
    u_list: numpy.array
        The set of knots
    """
    return nurbs(p_list, u_list, np.ones(len(p_list)))


def nurbs_surf(p_matrix, w_matrix, u_list, v_list):
    """Builds NURBS surface from OCC with numpy arrays

    Parameters
    ----------
    p_matrix: numpy.array
        Matrix of control points
    w_matrix: numpy.array
        Matrix of weights
    u_list, v_list: numpy.array
        Knots vectors
    """
    # Split knots vector into unique values and multiplicities
    u_val, u_mul = np.unique(u_list, return_counts=True)
    v_val, v_mul = np.unique(v_list, return_counts=True)
    # Basis degrees
    n, m, d = p_matrix.shape
    p = len(u_list) - n - 1
    q = len(v_list) - m - 1
    # OCC Arrays
    poles = np2occ_points(p_matrix)
    weights = np2occ_auto(w_matrix)
    u_knots = np2occ_auto(u_val)
    u_mults = np2occ_auto(u_mul)
    v_knots = np2occ_auto(v_val)
    v_mults = np2occ_auto(v_mul)
    return Geom_BSplineSurface(poles, weights, u_knots, v_knots, u_mults, v_mults, p, q)


def bspline_surf(p_matrix, u_list, v_list):
    """Builds B-Spline surface from OCC with numpy arrays

        Parameters
        ----------
        p_matrix: numpy.array
            Matrix of control points
        u_list, v_list: numpy.array
            Knots vectors
        """
    return nurbs_surf(p_matrix, np.ones((len(p_matrix), len(p_matrix[0]))), u_list, v_list)
