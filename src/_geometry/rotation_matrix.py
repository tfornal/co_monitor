import numpy as np


def rotation_matrix(theta, axis) -> np.array:
    """
    Returns the rotation matrix for a given angle `theta` and rotation `axis`.

    Parameters
    ----------
    theta : float
        The angle of rotation, in radians.
    axis : array_like
        The axis of rotation, given as a 1D array of 3 elements.

    Returns
    -------
    rot_matrix : ndarray, shape (3, 3)
        The rotation matrix.
    """
    axis = np.asarray(axis) / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a**2, b**2, c**2, d**2
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    rot_matrix = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )

    return rot_matrix
