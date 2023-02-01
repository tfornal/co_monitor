"""
Rotation matrix based on the Euler-Rodrigues formula for matrix conversion of 3d object.
Input - angle (in radians), axis in carthesian coordinates [x,y,z].
Return - rotated  matrix.
"""

import numpy as np

def rotation_matrix(theta, axis):

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