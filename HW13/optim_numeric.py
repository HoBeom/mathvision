import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.autograd.functional import hessian
# from math import sin

def f(x, y):
    return np.sin(x + y - 1) + (x - y - 1) **2 - 1.5 * x + 2.5 * y + 1

def torch_func(x, y):
    return torch.sin(x + y - 1) + (x - y - 1) **2 - 1.5 * x + 2.5 * y + 1

def gradient(x, y):
    cos_term = np.cos(-x - y + 1)
    x_grad = cos_term + 2 * x - 2 * y - 3.5
    y_grad = cos_term - 2 * x + 2 * y + 4.5 
    return np.array([x_grad, y_grad])

# plot f(x, y) = (x+y)(xy+xy^2)
x = np.linspace(-1, 5, 100)
y = np.linspace(-3, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
plt.show()

# plot 3d f(x, y) = (x+y)(xy+xy^2)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap="RdBu_r")
plt.show()

# plot gradient field
plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
grad = gradient(X, Y)
plt.quiver(X, Y, grad[0], grad[1], color="red", scale=20)
plt.show()

# gradient descent find local minimum test
# p = np.array([1.0, 3.0])
# pathway = []
# lr = 0.01
# for i in range(200):
#     grad = gradient(p[0], p[1])
#     p -= lr * grad
#     if i % 5 == 0:
#         pathway.append(p.copy())
# pathway = np.array(pathway)
# # print(f"gradient descent: {pathway}")
# plt.contour(X, Y, Z, levels=200, cmap="RdBu_r", alpha=0.5)
# plt.scatter(pathway[0, 0], pathway[0, 1], color="red")
# plt.scatter(pathway[-1, 0], pathway[-1, 1], color="red")
# plt.quiver(pathway[:-1, 0], pathway[:-1, 1], pathway[1:, 0] - pathway[:-1, 0], pathway[1:, 1] - pathway[:-1, 1], color="red", scale=10)
# plt.show()
# p = np.array([1.0, 3.0])

def gradient_descent(start_point, lr=0.01, max_iter=3000):
    # gradient descent until convergence
    print(f"gradient descent start point {start_point}")
    pathway = []
    p = np.array(start_point)
    prep = p.copy()
    for i in range(max_iter):
        grad = gradient(p[0], p[1])
        p -= lr * grad
        if i % 5 == 0:
            pathway.append(p.copy())
        if np.linalg.norm(p - prep) < 1e-8:
            print(f"gradient descent converge at {i} steps")
            break
        prep = p.copy()
    pathway = np.array(pathway)
    # print(f"gradient descent: {pathway}")
    print(f"gradient descent converge point {pathway[-1]}")
    plt.contour(X, Y, Z, levels=200, cmap="RdBu_r", alpha=0.5)
    plt.scatter(pathway[0, 0], pathway[0, 1], color="red")
    plt.scatter(pathway[-1, 0], pathway[-1, 1], color="red")
    plt.quiver(pathway[:-1, 0], pathway[:-1, 1], pathway[1:, 0] - pathway[:-1, 0], pathway[1:, 1] - pathway[:-1, 1], color="red", scale=10)
    plt.show()

gradient_descent([1.0, 3.0])
gradient_descent([0.0, 2.0])
gradient_descent([4.0, -2.0])
gradient_descent([0.0, 3.0])
gradient_descent([0.0, -3.0])

# Newton's method
def newton_method(start_point, lr=0.01, max_iter=3000):
    # Newton's method until convergence
    print(f"Newton's method start point {start_point}")
    p = np.array(start_point)
    pathway = []
    prep = p.copy()
    for i in range(max_iter):
        grad = gradient(p[0], p[1])
        hess = hessian(torch_func, (torch.tensor(p[0]), torch.tensor(p[1])))
        p -= lr * np.linalg.inv(hess) @ grad
        if i % 10 == 0:
            pathway.append(p.copy())
        if np.linalg.norm(p - prep) < 1e-8:
            print(f"Newton's method converge at {i} steps")
            break
        prep = p.copy()
    pathway = np.array(pathway)
    # print(f"Newton's method: {pathway}")
    print(f"Newton's method converge point {pathway[-1]}")
    plt.contour(X, Y, Z, levels=200, cmap="RdBu_r", alpha=0.5)
    plt.scatter(pathway[0, 0], pathway[0, 1], color="red")
    plt.scatter(pathway[-1, 0], pathway[-1, 1], color="red")
    plt.quiver(pathway[:-1, 0], pathway[:-1, 1], pathway[1:, 0] - pathway[:-1, 0], pathway[1:, 1] - pathway[:-1, 1], color="red", scale=10)
    plt.show()

newton_method([1.0, 3.0])
newton_method([0.0, 2.0])
newton_method([4.0, -2.0])
newton_method([0.0, 3.0])
newton_method([0.0, -3.0])

# saddle-free newton's method
def saddle_free_newton_method(start_point, lr=0.01, max_iter=10000):
    # saddle-free Newton's method until convergence
    print(f"saddle-free Newton's method start point {start_point}")
    p = np.array(start_point)
    pathway = []
    prep = p.copy()
    for i in range(max_iter):
        grad = gradient(p[0], p[1])
        hess = hessian(torch_func, (torch.tensor(p[0]), torch.tensor(p[1])))
        w, v = np.linalg.eig(hess)
        d = np.diag(np.abs(w))
        abs_hess = v.T @ d @ v # absolute value of hessian
        p -= lr * np.linalg.inv(abs_hess) @ grad
        if i % 10 == 0:
            pathway.append(p.copy())
        if np.linalg.norm(p - prep) < 1e-8:
            print(f"saddle-free Newton's method converge at {i} steps")
            break
        prep = p.copy()
    pathway = np.array(pathway)
    # print(f"saddle-free Newton's method: {pathway}")
    print(f"saddle-free Newton's method converge point {pathway[-1]}")
    plt.contour(X, Y, Z, levels=200, cmap="RdBu_r", alpha=0.5)
    plt.scatter(pathway[0, 0], pathway[0, 1], color="red")
    plt.scatter(pathway[-1, 0], pathway[-1, 1], color="red")
    plt.quiver(pathway[:-1, 0], pathway[:-1, 1], pathway[1:, 0] - pathway[:-1, 0], pathway[1:, 1] - pathway[:-1, 1], color="red", scale=10)
    plt.show()

saddle_free_newton_method([1.0, 3.0])
saddle_free_newton_method([0.0, 2.0])
saddle_free_newton_method([4.0, -2.0])
saddle_free_newton_method([0.0, 3.0])
saddle_free_newton_method([0.0, -3.0])

# Axes3D
from mpl_toolkits.mplot3d import Axes3D
def newton_method_3d(start_point, lr=0.01, max_iter=3000):
    print(f"Newton's method start point {start_point}")
    p = np.array(start_point)
    pathway = []
    prep = p.copy()
    for i in range(max_iter):
        grad = gradient(p[0], p[1])
        hess = hessian(torch_func, (torch.tensor(p[0]), torch.tensor(p[1])))
        p -= lr * np.linalg.inv(hess) @ grad
        if i % 10 == 0:
            pathway.append(p.copy())
        if np.linalg.norm(p - prep) < 1e-8:
            print(f"Newton's method converge at {i} steps")
            break
        prep = p.copy()
    pathway = np.array(pathway)
    # print(f"Newton's method: {pathway}")
    print(f"Newton's method converge point {pathway[-1]}")
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow", alpha=0.5)
    ax.plot(pathway[:, 0], pathway[:, 1], torch_func(torch.tensor(pathway[:, 0]), torch.tensor(pathway[:, 1])).numpy(), color="red")
    ax.scatter(pathway[0, 0], pathway[0, 1], torch_func(torch.tensor(pathway[0, 0]), torch.tensor(pathway[0, 1])).numpy(), color="red")
    ax.scatter(pathway[-1, 0], pathway[-1, 1], torch_func(torch.tensor(pathway[-1, 0]), torch.tensor(pathway[-1, 1])).numpy(), color="red")
    plt.show()

def saddle_free_newton_method_3d(start_point, lr=0.01, max_iter=3000):
    print(f"saddle-free Newton's method start point {start_point}")
    p = np.array(start_point)
    pathway = []
    prep = p.copy()
    for i in range(max_iter):
        grad = gradient(p[0], p[1])
        hess = hessian(torch_func, (torch.tensor(p[0]), torch.tensor(p[1])))
        w, v = np.linalg.eig(hess)
        d = np.diag(np.abs(w))
        abs_hess = v.T @ d @ v # absolute value of hessian
        p -= lr * np.linalg.inv(abs_hess) @ grad
        if i % 10 == 0:
            pathway.append(p.copy())
        if np.linalg.norm(p - prep) < 1e-8:
            print(f"saddle-free Newton's method converge at {i} steps")
            break
        prep = p.copy()
    pathway = np.array(pathway)
    # print(f"saddle-free Newton's method: {pathway}")
    print(f"saddle-free Newton's method converge point {pathway[-1]}")
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow", alpha=0.5)
    ax.plot(pathway[:, 0], pathway[:, 1], torch_func(torch.tensor(pathway[:, 0]), torch.tensor(pathway[:, 1])).numpy(), color="red")
    ax.scatter(pathway[0, 0], pathway[0, 1], torch_func(torch.tensor(pathway[0, 0]), torch.tensor(pathway[0, 1])).numpy(), color="red")
    ax.scatter(pathway[-1, 0], pathway[-1, 1], torch_func(torch.tensor(pathway[-1, 0]), torch.tensor(pathway[-1, 1])).numpy(), color="red")
    plt.show()


newton_method_3d([1.0, 3.0])
saddle_free_newton_method_3d([1.0, 3.0])
newton_method_3d([0.0, 2.0])
saddle_free_newton_method_3d([0.0, 2.0])
newton_method_3d([4.0, -2.0])
saddle_free_newton_method_3d([4.0, -2.0])
newton_method_3d([0.0, 3.0])
saddle_free_newton_method_3d([0.0, 3.0])
newton_method_3d([0.0, -3.0])
saddle_free_newton_method_3d([0.0, -3.0])

