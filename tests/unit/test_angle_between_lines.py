import numpy as np
import pytest


def test_angle_between_lines():
    central_point = [0, 0, 0]
    p1 = [1, 0, 0]
    p2 = [0, 1, 0]
    expected_angle = 90.0

    angle = angle_between_lines(central_point, p1, p2)
    assert np.isclose(angle, expected_angle, rtol=1e-5)

    p1 = [1, 1, 0]
    p2 = [0, 1, 0]
    expected_angle = 45.0

    angle = angle_between_lines(central_point, p1, p2)
    assert np.isclose(angle, expected_angle, rtol=1e-5)