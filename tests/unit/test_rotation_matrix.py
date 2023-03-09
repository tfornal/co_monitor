import numpy as np
import pytest

from co_monitor.geometry.rotation_matrix import rotation_matrix
from co_monitor.geometry.mesh_calculation import CuboidMesh


def test_rotation_matrix_90_degrees_around_x_axis():
    theta = np.pi / 2
    axis = [1, 0, 0]
    rot_matrix = rotation_matrix(theta, axis)

    expected_rot_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    np.testing.assert_allclose(rot_matrix, expected_rot_matrix)


@pytest.mark.skip()
def test_rotation_matrix_180_degrees_around_y_axis():
    theta = np.pi
    axis = [0, 1, 0]
    rot_matrix = rotation_matrix(theta, axis)

    expected_rot_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    np.testing.assert_allclose(rot_matrix, expected_rot_matrix)


def test_rotation_matrix_45_degrees_around_z_axis():
    theta = np.pi / 4
    axis = [0, 0, 1]
    rot_matrix = rotation_matrix(theta, axis)

    expected_rot_matrix = np.array(
        [
            [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0],
            [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
            [0, 0, 1],
        ]
    )

    np.testing.assert_allclose(rot_matrix, expected_rot_matrix)
