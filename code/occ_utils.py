import numpy as np
from OCC.Core.Geom import Geom_BSplineCurve
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt


def np2occ_points(points):
    """Converts numpy array of points into occ points array
    """
    n, m = points.shape
    occ_array = TColgp_Array1OfPnt(1, n)
    for i in range(n):
        if m == 2:
            occ_array.SetValue(i + 1, gp_Pnt(points[i, 0], points[i, 1], 0.))
        elif m == 3:
            occ_array.SetValue(i + 1, gp_Pnt(points[i, 0], points[i, 1], points[i, 2]))
    return occ_array


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
    array = container(1, len(data))
    for i in range(len(data)):
        array.SetValue(i + 1, cast(data[i]))
    return array


def np2occ_auto(v):
    """Converts automatic numpy array into a occ array in base of dtype"""
    data_type = v.dtype
    if data_type == np.dtype('float64'):
        return np2occ(v, TColStd_Array1OfReal, float)
    elif data_type == np.dtype('int64'):
        return np2occ(v, TColStd_Array1OfInteger, int)
    else:
        raise ValueError('data type not supported')


def bspline(p_list, t_list):
    """Construct B-spline curve from OCC with numpy arrays

    Parameters
    ----------
    p_list: numpy.array
        The set of control points
    t_list: numpy.array
        The set of knots
    """
    t_u, t_m = np.unique(t_list, return_counts=True)
    k = len(t_list) - len(p_list) - 1  # degree
    poles = np2occ_points(p_list)
    knots = np2occ_auto(t_u)
    mults = np2occ_auto(t_m)
    return Geom_BSplineCurve(poles, knots, mults, k)


def nurbs(p_list, t_list, w_list):
    """Construct NURBS curve from OCC with numpy arrays

    Parameters
    ----------
    p_list: numpy.array
        The set of control points
    t_list: numpy.array
        The knots vector
    w_list: numpy.array
        The weighs vector
    """
    t_u, t_m = np.unique(t_list, return_counts=True)
    k = len(t_list) - len(p_list) - 1  # degree
    poles = np2occ_points(p_list)
    weighs = np2occ_auto(w_list)
    knots = np2occ_auto(t_u)
    mults = np2occ_auto(t_m)
    return Geom_BSplineCurve(poles, weighs, knots, mults, k)
