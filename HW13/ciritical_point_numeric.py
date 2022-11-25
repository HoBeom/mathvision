import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.autograd.functional import hessian

def f(x, y):
    # f(x, y) = (x+y)(xy+xy^2)
    return (x + y) * (x * y + x * y * y)

def gradient(x, y):
    # f(x, y) = (x+y)(x*y+x*y^2)
    # df/dx = y(y+1)(2x+y)
    # df/dy = x(2xy+x+y(3y+2)) 
    # return np.array([y * (y + 1) * (2 * x + y), x * (2 * x * y + x + y * (3 * y + 2))])
    # x*y**2 + x*y + (x + y)*(y**2 + y)
    # x*y**2 + x*y + (x + y)*(2*x*y + x)
    ymy = y * y
    xmy = x * y
    xpy = x + y
    temp = x * ymy + xmy
    return np.array([temp + xpy * (ymy + y), temp + xpy * (2 * xmy + x)])

def hobe_hessian(x, y):
    # f(x, y) = (x+y)(x*y+x*y^2)
    # df/dx = y(y+1)(2x+y)
    # df/dy = x(2xy+x+y(3y+2))
    # d^2f/dx^2 = 2y(y+1)
    # d^2f/dy^2 = 2x(x+3y+1)
    # d^2f/dxdy = x(4y+2)+y(3y+2)
    d2f_dx2 = 2 * y * (y + 1)
    d2f_dy2 = 2 * x * (x + 3 * y + 1)
    d2f_dxdy = x * (4 * y + 2) + y * (3 * y + 2)
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

# plot f(x, y) = (x+y)(xy+xy^2)
x = np.linspace(-1, 1.5, 100)
y = np.linspace(-1.2, 0.2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
plt.show()

# plot 3d f(x, y) = (x+y)(xy+xy^2)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap="RdBu_r")
plt.show()

# initial point
p = np.array([-0.11616162, 0.2])
grad = gradient(p[0], p[1])
print(f"initial point: {p}, gradient: {grad}")
# plot gradient vector
plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
plt.quiver(p[0], p[1], grad[0], grad[1], color="red", scale=10)
plt.show()

# plot gradient field
plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
grad = gradient(X, Y)
plt.quiver(X, Y, grad[0], grad[1], color="red", scale=20)
plt.show()


# # hessian function test
# for i in x:
#     for j in y:
#         x_tensor = torch.tensor(i)
#         y_tensor = torch.tensor(j)
#         hess = hessian(f, (x_tensor, y_tensor))
#         hobe_hess = hobe_hessian(i,j)
#         assert hess[0][0].item() - hobe_hess[0][0] < 1e-10
#         assert hess[0][1].item() - hobe_hess[0][1] < 1e-10
#         assert hess[1][0].item() - hobe_hess[1][0] < 1e-10
#         assert hess[1][1].item() - hobe_hess[1][1] < 1e-10

# find critical point by meshgrid
zero_grad_points = np.all(np.abs(grad) < 1e-2, axis=0)
critical_points = np.array([X[zero_grad_points], Y[zero_grad_points]])
print(f"critical points: {critical_points}, shape: {critical_points.shape}")
plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
plt.scatter(critical_points[0], critical_points[1], color="red")
plt.show()


# hesse matrix
# for x, y in critical_points.T:
#     hess = hobe_hessian(x, y)
#     print(f"critical point: {np.array([x, y])}, hessian: {hess}")
#     eigenvalues, eigenvectors = np.linalg.eig(hess)
#     print(f"eigenvalues: {eigenvalues}, eigenvectors: {eigenvectors}")
#     if eigenvalues[0] > 0 and eigenvalues[1] > 0:
#         print(f"local minimum: {np.array([x, y])}")
#         color = "blue"
#     elif eigenvalues[0] < 0 and eigenvalues[1] < 0:
#         print(f"local maximum: {np.array([x, y])}")
#         color = "red"
#     else:
#         print(f"saddle point: {np.array([x, y])}")
#         color = "green"
#     # plot eigen vectors
#     plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
#     plt.scatter(x, y, color=color)
#     for i in range(eigenvectors.shape[1]):
#         plt.quiver(x, y, eigenvectors[0][i], eigenvectors[1][i], color=color, scale=10)
#     plt.show()

# gradient descent find local minimum
p = np.array([0.0, -1.0])
pathway = []
lr = 0.1
for i in range(100):
    grad = gradient(p[0], p[1])
    p -= lr * grad
    if i % 10 == 0:
        pathway.append(p.copy())
pathway = np.array(pathway)
print(f"gradient descent: {pathway}")
plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
plt.scatter(pathway[0, 0], pathway[0, 1], color="red")
plt.scatter(pathway[-1, 0], pathway[-1, 1], color="red")
plt.quiver(pathway[:-1, 0], pathway[:-1, 1], pathway[1:, 0] - pathway[:-1, 0], pathway[1:, 1] - pathway[:-1, 1], color="red", scale=5)
plt.show()


# gradient descent find local maximum
p = np.array([0.0, -1.0])
pathway = []
lr = 0.1
for i in range(100):
    grad = gradient(p[0], p[1])
    p += lr * grad
    if i % 10 == 0:
        pathway.append(p.copy())
pathway = np.array(pathway)
print(f"gradient descent: {pathway}")
plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
plt.scatter(pathway[0, 0], pathway[0, 1], color="red")
plt.scatter(pathway[-1, 0], pathway[-1, 1], color="red")
plt.quiver(pathway[:-1, 0], pathway[:-1, 1], pathway[1:, 0] - pathway[:-1, 0], pathway[1:, 1] - pathway[:-1, 1], color="red", scale=5)
plt.show()

