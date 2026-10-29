import numpy as np

from meatpick.perception.detector import touches_border, touches_bottom


def _blob(rows, cols=(600, 700)):
    m = np.zeros((720, 1280), bool)
    m[rows[0] : rows[1], cols[0] : cols[1]] = True
    return m


def test_a_mask_a_few_pixels_short_of_the_edge_still_counts_as_cut_off():
    # YOLO masks stop short of the frame edge, so an exact last-row test would miss these
    assert touches_border(_blob((500, 716)))
    assert touches_bottom(_blob((500, 716)))
    assert touches_border(_blob((4, 200)))


def test_a_mask_well_inside_the_frame_is_not_cut_off():
    assert not touches_border(_blob((300, 400)))
    assert not touches_bottom(_blob((300, 400)))


def test_side_edges_count_for_border_but_not_for_the_exit_test():
    side = _blob((300, 400), cols=(0, 100))
    assert touches_border(side) and not touches_bottom(side)
