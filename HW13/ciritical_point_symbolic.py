import numpy as np
from matplotlib import pyplot as plt
from sympy import symbols, Eq, solve

def f(x, y):
    # f(x, y) = (x+y)(xy+xy^2)
    return (x + y) * (x * y + x * y * y)

# https://towardsdatascience.com/hessian-matrix-and-optimization-problems-in-python-3-8-f7cd2a615371
def partial(element, function):
    """
    partial : sympy.core.symbol.Symbol * sympy.core.add.Add -> sympy.core.add.Add
    partial(element, function) Performs partial derivative of a function of several variables is its derivative with respect to one of those variables, with the others held constant. Return partial_diff.
    """
    partial_diff = function.diff(element)

    return partial_diff


def gradient_to_zero(symbols_list, partials):
    """
    gradient_to_zero : List[sympy.core.symbol.Symbol] * List[sympy.core.add.Add] -> Dict[sympy.core.numbers.Float]
    gradient_to_zero(symbols_list, partials) Solve the null equation for each variable, and determine the pair of coordinates of the singular point. Return singular.
    """
    partial_x = Eq(partials[0], 0)
    partial_y = Eq(partials[1], 0)

    sol = solve((partial_x, partial_y), (symbols_list[0], symbols_list[1]))
    print("Singular point is : {0}".format(sol))
    return sol

def get_critical_points():
    """
    Fonction principale.
    """
    x, y = symbols('x y', real=True)
    symbols_list = [x, y]
    function = (x + y) * (x * y + x * y * y)
    partials, partials_second = [], []

    for element in symbols_list:
        partial_diff = partial(element, function)
        partials.append(partial_diff)

    # grad = gradient(partials)
    singular = gradient_to_zero(symbols_list, partials)
    return singular

def gradient(x, y):
    ymy = y * y
    xmy = x * y
    xpy = x + y
    temp = x * ymy + xmy
    return np.array([temp + xpy * (ymy + y), temp + xpy * (2 * xmy + x)])

def hobe_hessian(x, y):
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

# plot gradient field
plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
grad = gradient(X, Y)
plt.quiver(X, Y, grad[0], grad[1], color="red", scale=40)
plt.show()

# get critical_point with sympy
critical_points = get_critical_points()
critical_points = np.array(critical_points, dtype=np.float64)
print(f"Critical points: {critical_points}")
# plot critical points
plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
plt.scatter(critical_points[:, 0], critical_points[:, 1], color="red")
plt.show()


# plot eigen vectors
for x, y in critical_points:
    plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
    print(f"Critical point: {x, y}")
    hess = hobe_hessian(x, y)
    print(f"critical point: {np.array([x, y])}, hessian: {hess}")
    eigenvalues, eigenvectors = np.linalg.eig(hess)
    print(f"eigenvalues: {eigenvalues}, eigenvectors: {eigenvectors}")
    if eigenvalues[0] > 0 and eigenvalues[1] > 0:
        print(f"local minimum: {np.array([x, y])}")
        color = "blue"
    elif eigenvalues[0] < 0 and eigenvalues[1] < 0:
        print(f"local maximum: {np.array([x, y])}")
        color = "red"
    else:
        print(f"saddle point: {np.array([x, y])}")
        color = "green"
    for i in range(eigenvectors.shape[1]):
        plt.scatter(x, y, color=color)
        plt.quiver(x, y, eigenvectors[0][i], eigenvectors[1][i], color=color, scale=10)
    plt.show()

# plot eigen vectors
plt.contour(X, Y, Z, levels=200, cmap="RdBu_r")
for x, y in critical_points:
    print(f"Critical point: {x, y}")
    hess = hobe_hessian(x, y)
    print(f"critical point: {np.array([x, y])}, hessian: {hess}")
    eigenvalues, eigenvectors = np.linalg.eig(hess)
    print(f"eigenvalues: {eigenvalues}, eigenvectors: {eigenvectors}")
    if eigenvalues[0] > 0 and eigenvalues[1] > 0:
        print(f"local minimum: {np.array([x, y])}")
        color = "blue"
    elif eigenvalues[0] < 0 and eigenvalues[1] < 0:
        print(f"local maximum: {np.array([x, y])}")
        color = "red"
    else:
        print(f"saddle point: {np.array([x, y])}")
        color = "green"
    for i in range(eigenvectors.shape[1]):
        plt.scatter(x, y, color=color)
        plt.quiver(x, y, eigenvectors[0][i], eigenvectors[1][i], color=color, scale=10)
plt.show()
