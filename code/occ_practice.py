from OCC.Display.SimpleGui import init_display

import examples_data
import occ_utils

display, start_display, _, _ = init_display()


def bspline_example(n):
    examples = examples_data.bspline()
    p_list = examples[n].points
    u_list = examples[n].knots
    return occ_utils.bspline(p_list, u_list)


def nurbs_example(n):
    examples = examples_data.nurbs()
    p_list = examples[n].points
    u_list = examples[n].knots
    w_list = examples[n].weights
    return occ_utils.nurbs(p_list, u_list, w_list)


if __name__ == '__main__':
    # B-spline example
    spline = bspline_example(3)

    # NURBS example
    # spline = nurbs_example(0)

    # Display spline
    display.DisplayShape(spline, update=True)
    start_display()
