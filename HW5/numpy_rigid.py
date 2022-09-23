import numpy as np


points_3d = np.array([[-0.500000, 0.000000, 2.121320], 
            [0.500000, 0.000000, 2.121320], 
            [0.500000, -0.707107, 2.828427]], dtype=np.float64)
points_3d_transformed = np.array([[1.363005, -0.427130, 2.339082], 
                        [1.748084, 0.437983, 2.017688], 
                        [2.636461, 0.184843, 2.400710]], dtype=np.float64)

def R3D(u: np.ndarray, c: float):
    s = np.sqrt(1 - c ** 2)
    transform = np.asarray([
        [c + u[0] ** 2 * (1 - c), u[0] * u[1] * (1 - c) - u[2] * s,
         u[0] * u[2] * (1 - c) + u[1] * s],
        [u[1] * u[0] * (1 - c) + u[2] * s, c + u[1] ** 2 * (1 - c),
         u[1] * u[2] * (1 - c) - u[0] * s],
        [u[2] * u[0] * (1 - c) - u[1] * s, u[2] * u[1] * (1 - c) + u[0] * s,
         c + u[2] ** 2 * (1 - c)]
    ])
    return transform

def get_normal_vector(a, b):
    v = np.cross(a, b)
    return v / np.linalg.norm(v)

def get_cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def getRigidTransform(A, B):
    # get translation vector
    Center_A = A[0]
    Center_B = B[0]
    A = A - Center_A + Center_B
    # get normal vector of plane A and B
    nA = get_normal_vector(A[1] - A[0], A[2] - A[0])
    nB = get_normal_vector(B[1] - B[0], B[2] - B[0])
    # get vector h
    h = get_normal_vector(nA, nB)
    cos = get_cos(nA, nB)
    R1 = R3D(h, cos)
    p1p3_rotated = np.matmul(R1, A[2] - A[0])
    cos = get_cos(p1p3_rotated, B[2] - B[0])
    R2 = R3D(nB, cos)
    return R1, R2, Center_A, Center_B

R1, R2, Center_A, Center_B = getRigidTransform(points_3d, points_3d_transformed)

test_sample = [[-0.500000, 0.000000, 2.121320], 
                [0.500000, 0.000000, 2.121320], 
                [0.500000, -0.707107, 2.828427], 
                [0.500000, 0.707107, 2.828427], 
                [1,1,1]]

for X in test_sample:
    print(np.matmul(np.matmul(R1,(X - Center_A)), R2) + Center_B)

