import numpy as np


def test_rotation_matrix(rot):
    if rot.ndim != 2 or rot.shape[0] != rot.shape[1]:
        return False
    is_identity = np.allclose(rot.dot(rot.T), 
                                     np.identity(rot.shape[0], float))
    is_one = np.allclose(np.linalg.det(rot), 1)
    
    return is_identity and is_one 


if __name__ == '__main__':
    rot = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ])
    print(test_rotation_matrix(rot))  # True
    
    
    rot = np.array([
        [-0.05055035,  0.06100042,  0.99685687],
        [-0.06100042,  0.99608083, -0.06404625],
        [-0.99685687, -0.06404625, -0.04663118]])
    print(test_rotation_matrix(rot))  # True
    
    
    rot = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    print(test_rotation_matrix(rot))  # False
    print(test_rotation_matrix(np.zeros((3, 2))))  # False