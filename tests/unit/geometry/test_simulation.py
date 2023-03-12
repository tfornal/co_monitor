import numpy as np
import pytest
import os
import dask.array as da

from co_monitor.geometry.simulation import Simulation


@pytest.mark.skip(reason="cannot test this method yet")
def test_angle_between_lines():
    calc = Simulation.calculate_angles()

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
