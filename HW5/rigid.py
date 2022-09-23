import torch

try:
    from rich import print

    print("print() replaced with rich.print()")
except ImportError:
    pass


class RigidTransform:
    def __init__(self, A, B):
        if type(A) != torch.Tensor or type(B) != torch.Tensor:
            A = torch.tensor(A, dtype=torch.float64)
            B = torch.tensor(B, dtype=torch.float64)
        self.setRigidTransform(A, B)

    def get_normal_vector(self, a, b):
        v = torch.cross(a, b)
        return v / torch.linalg.norm(v)

    def get_cos(self, a, b):
        return torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))

    def R3D(self, u: torch.tensor, c: float):
        s = torch.sqrt(1 - c**2)
        matrix = torch.tensor(
            [
                [
                    c + u[0] ** 2 * (1 - c),
                    u[0] * u[1] * (1 - c) - u[2] * s,
                    u[0] * u[2] * (1 - c) + u[1] * s,
                ],
                [
                    u[1] * u[0] * (1 - c) + u[2] * s,
                    c + u[1] ** 2 * (1 - c),
                    u[1] * u[2] * (1 - c) - u[0] * s,
                ],
                [
                    u[2] * u[0] * (1 - c) - u[1] * s,
                    u[2] * u[1] * (1 - c) + u[0] * s,
                    c + u[2] ** 2 * (1 - c),
                ],
            ]
        )
        return matrix

    def setRigidTransform(self, A, B):
        # A, B: torch.tensor
        # Set center of A and B
        self.Center_A = A[0]
        self.Center_B = B[0]
        # translate A to B
        A = A - self.Center_A + self.Center_B
        # get normal vector of plane A and B
        nA = self.get_normal_vector(A[1] - A[0], A[2] - A[0])
        nB = self.get_normal_vector(B[1] - B[0], B[2] - B[0])
        # get vector h and cos
        h1 = self.get_normal_vector(nA, nB)
        cos1 = self.get_cos(nA, nB)
        # calculate R1 
        self.R1 = self.R3D(h1, cos1)
        # get vector h2 and cos2
        h2 = self.R1 @ (A[2] - A[0])
        cos2 = self.get_cos(h2, B[2] - B[0])
        # calculate R2
        self.R2 = self.R3D(nB, cos2)


    def __call__(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float64)
        if len(x.shape) == 1:
            return ((self.R1 @ (x - self.Center_A)) @ self.R2) + self.Center_B
        else:
            return ((self.R1 @ (x - self.Center_A).T).T @ self.R2) + self.Center_B


if __name__ == "__main__":

    points_3d = torch.tensor(
        [
            [-0.500000, 0.000000, 2.121320],
            [0.500000, 0.000000, 2.121320],
            [0.500000, -0.707107, 2.828427],
        ],
        dtype=torch.float64,
    )
    points_3d_transformed = torch.tensor(
        [
            [1.363005, -0.427130, 2.339082],
            [1.748084, 0.437983, 2.017688],
            [2.636461, 0.184843, 2.400710],
        ],
        dtype=torch.float64,
    )

    rigid_transform = RigidTransform(points_3d, points_3d_transformed)

    test_samples = [
        [-0.500000, 0.000000, 2.121320],
        [0.500000, 0.000000, 2.121320],
        [0.500000, -0.707107, 2.828427],
        [0.500000, 0.707107, 2.828427],
        [1, 1, 1],
    ]

    for sample in test_samples:
        print(rigid_transform(sample))

    print(rigid_transform(test_samples))
