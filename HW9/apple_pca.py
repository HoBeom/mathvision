from pickletools import markobject
import numpy as np
import matplotlib.pyplot as plt

# file open
with open('data_a.txt', 'r') as f:
    data_a = f.read().splitlines()
data_a = [i.split(',') for i in data_a]
data_a = np.array(data_a, dtype=np.float32)
with open('data_b.txt', 'r') as f:
    data_b = f.read().splitlines()
data_b = [i.split(',') for i in data_b]
data_b = np.array(data_b, dtype=np.float32)

# get mean and covariance
def get_mean_cov(data):
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mean, cov

# get eigenvalue and eigenvector
def get_pca(data):
    mean, cov = get_mean_cov(data)
    # get eigenvalue and eigenvector
    eigenvalue, eigenvector = np.linalg.eig(cov)
    # sort eigenvalue and eigenvector
    idx = eigenvalue.argsort()[::-1]
    eigenvalue = eigenvalue[idx]
    eigenvector = eigenvector[:, idx]
    return mean, cov, eigenvalue, eigenvector

# get projection on eigenvector
def projection_on_eigenvector(data, mean, eigenvector, top_k=2):
    return np.dot(data - mean, eigenvector[:, :top_k])

# get pca principal component and eigenvalue of A, B
mean_a, cov_a, eigenvalue_a, eigenvector_a = get_pca(data_a)
mean_b, cov_b, eigenvalue_b, eigenvector_b = get_pca(data_b)
mean_ab, cov_ab, eigenvalue_ab, eigenvector_ab = get_pca(np.concatenate((data_a, data_b), axis=0))
print(f"mean_a:{mean_a}, cov_a:{cov_a}, eigenvalue_a:{eigenvalue_a}, eigenvector_a:{eigenvector_a}")
print(f"mean_b:{mean_b}, cov_b:{cov_b}, eigenvalue_b:{eigenvalue_b}, eigenvector_b:{eigenvector_b}")
print(f"mean_ab:{mean_ab}, cov_ab:{cov_ab}, eigenvalue_ab:{eigenvalue_ab}, eigenvector_ab:{eigenvector_ab}")


# get projection of A, B
projection_a = projection_on_eigenvector(data_a, mean_ab, eigenvector_ab)
projection_b = projection_on_eigenvector(data_b, mean_ab, eigenvector_ab)

# plt visualization projection 2D using first 2 principal component of A, B
plt.scatter(projection_a[:, 0], projection_a[:, 1], c='r', label='A')
plt.scatter(projection_b[:, 0], projection_b[:, 1], c='b', label='B')
plt.legend()
plt.show()

projection_mean_a, projection_cov_a = get_mean_cov(projection_a)
projection_mean_b, projection_cov_b = get_mean_cov(projection_b)

import tensorflow_probability as tfp
tfd = tfp.distributions
len_a = len(data_a)
len_b = len(data_b)
len_total = len_a + len_b
pi = np.array([len_a / len_total, len_b / len_total])
mu = np.array([projection_mean_a, projection_mean_b])
sigma = np.array([projection_cov_a, projection_cov_b])
print(f"pi:{pi}, mu:{mu}, sigma:{sigma}")
gmm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=pi),
    components_distribution=tfd.MultivariateNormalTriL(
        loc=mu, scale_tril=np.linalg.cholesky(sigma)))

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
x, y = np.meshgrid(x, y)
prob = gmm.prob(np.stack([x, y], axis=-1))
print(f'prob.shape:{prob.shape}')
ax = plt.axes(projection='3d')
plt.contour(x, y, prob.numpy(), 10)
ax.plot_surface(x, y, prob, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
plt.show()


# load test data
with open('test.txt', 'r') as f:
    test = f.read().splitlines()
test = [i.split(',') for i in test]
test = np.array(test, dtype=np.float32)


# get projection of test
projection_test = projection_on_eigenvector(test, mean_ab, eigenvector_ab)
plt.scatter(projection_a[:, 0], projection_a[:, 1], c='r', label='A', alpha=0.5)
plt.scatter(projection_b[:, 0], projection_b[:, 1], c='b', label='B', alpha=0.5)
plt.scatter(projection_test[:, 0], projection_test[:, 1], c='g', label='Test')
plt.legend()
plt.show()

def get_mahalanobis_distance(x, mean, cov):
    S_half = np.linalg.cholesky(np.linalg.inv(cov))
    return np.linalg.norm(np.dot(S_half, (x - mean).T), axis=0)

# projection mahalanobis distance of test
mahalanobis_distance_a = np.array([
    get_mahalanobis_distance(i, projection_mean_a, projection_cov_a) 
    for i in projection_test])
mahalanobis_distance_b = np.array([
    get_mahalanobis_distance(i, projection_mean_b, projection_cov_b) 
    for i in projection_test])
print(f'mahalanobis_distance_a:{mahalanobis_distance_a}')
print(f'mahalanobis_distance_b:{mahalanobis_distance_b}')

# concat mahalanobis distance of test
mahalanobis_distance = np.concatenate((mahalanobis_distance_a.reshape(-1, 1), mahalanobis_distance_b.reshape(-1, 1)), axis=1)
print(f'mahalanobis_distance:{mahalanobis_distance}')

# classification of test
classification = np.argmax(-mahalanobis_distance, axis=1)
print(f'classification:{classification}')

# visualization classification of test
plt.scatter(projection_a[:, 0], projection_a[:, 1], c='r', label='A', alpha=0.2)
plt.scatter(projection_b[:, 0], projection_b[:, 1], c='b', label='B', alpha=0.2)
plt.scatter(projection_test[classification == 0, 0], projection_test[classification == 0, 1], c='g', label='Predict_A', marker='s')
plt.scatter(projection_test[classification == 1, 0], projection_test[classification == 1, 1], c='orange', label='Predict_B', marker='x')
plt.legend()
plt.show()

# visualization classification of grid
min_projection_a = np.min(projection_a, axis=0)
max_projection_a = np.max(projection_a, axis=0)
min_projection_b = np.min(projection_b, axis=0)
max_projection_b = np.max(projection_b, axis=0)
min_projection = np.min(np.array([min_projection_a, min_projection_b]), axis=0)
max_projection = np.max(np.array([max_projection_a, max_projection_b]), axis=0)
x = np.linspace(min_projection[0], max_projection[0], 100)
y = np.linspace(min_projection[1], max_projection[1], 100)
x, y = np.meshgrid(x, y)
grid = np.stack([x, y], axis=-1)
grid = grid.reshape(-1, 2)

grid_mahalanobis_distance_a = np.array([get_mahalanobis_distance(i, projection_mean_a, projection_cov_a) for i in grid])
grid_mahalanobis_distance_b = np.array([get_mahalanobis_distance(i, projection_mean_b, projection_cov_b) for i in grid])
grid_mahalanobis_distance = np.concatenate((grid_mahalanobis_distance_a.reshape(-1, 1), grid_mahalanobis_distance_b.reshape(-1, 1)), axis=1)
grid_classification = np.argmax(-grid_mahalanobis_distance, axis=1)

plt.scatter(grid[grid_classification == 0, 0], grid[grid_classification == 0, 1], c='g', label='Predict_A', marker='s', alpha=0.2)
plt.scatter(grid[grid_classification == 1, 0], grid[grid_classification == 1, 1], c='orange', label='Predict_B', marker='x', alpha=0.2)
plt.scatter(projection_a[:, 0], projection_a[:, 1], c='r', label='A')
plt.scatter(projection_b[:, 0], projection_b[:, 1], c='b', label='B')

plt.legend()
plt.show()








