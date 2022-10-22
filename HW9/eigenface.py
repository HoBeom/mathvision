import numpy as np
import cv2
from glob import glob

def load_images(path):
    images = []
    paths = glob(path)
    for file in paths:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    labels = [i.split('/')[-1].split('.')[0] for i in paths]
    labels = [int(i.split('_')[0].replace('s',''))*10 + int(i.split('_')[1])-1 for i in labels]
    idx = np.argsort(labels)
    images = np.array(images)[idx]
    labels = np.array(labels)[idx]
    return images, labels

def split_data(images, labels):
    train_images = [images[i] for i in range(len(labels)) if labels[i] % 10 != 0] # 0~9
    train_label = [labels[i] for i in range(len(labels)) if labels[i] % 10 != 0]
    test_images = [images[i] for i in range(len(labels)) if labels[i] % 10 == 0]
    test_label = [labels[i] for i in range(len(labels)) if labels[i] % 10 == 0]
    return np.array(train_images), np.array(train_label), np.array(test_images), np.array(test_label)

def get_mean_cov(data):
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mean, cov

def get_pca(data):
    mean, cov = get_mean_cov(data)
    # get eigenvalue and eigenvector
    eigenvalue, eigenvector = np.linalg.eig(cov)
    # sort eigenvalue and eigenvector
    idx = eigenvalue.argsort()[::-1]
    eigenvalue = eigenvalue[idx]
    eigenvector = eigenvector[:, idx]
    return mean, cov, eigenvalue, eigenvector

def projection_on_eigenvector(data, mean, eigenvector, top_k=2):
    return np.dot(data - mean, eigenvector[:, :top_k])

def get_projection(images, mean, eigenvector, top_k=2):
    projection = []
    for img in images:
        img = img.flatten()
        projection.append(projection_on_eigenvector(img, mean, eigenvector, top_k))
    return np.array(projection)

def get_person_mean(train_projection, train_person_label):
    person_mean = []
    for id in range(1,41):
        person_mean.append(np.mean(train_projection[train_person_label == id], axis=0))
    return np.array(person_mean)

def get_person_cov(train_projection, train_person_label):
    person_cov = []
    for id in range(1,41):
        person_cov.append(np.cov(train_projection[train_person_label == id], rowvar=False))
    return np.array(person_cov)
    
def mahalanobis_distance(x, mean, cov):
    S_half = np.linalg.cholesky(np.linalg.inv(cov))
    return np.linalg.norm(np.dot(S_half, (x - mean).T), axis=0)

def mahalanobis_distance_matrix(test_projection, train_projection, train_person_label):
    person_cov = get_person_cov(train_projection, train_person_label)
    person_mean = get_person_mean(train_projection, train_person_label)
    distance = []
    for i in range(len(test_projection)):
        distance.append([])
        for j in range(len(person_mean)):
            distance[i].append(mahalanobis_distance(test_projection[i], person_mean[j], person_cov[j]))
    return np.array(distance)

def euclidean_distance(x, mean):
    x2 = np.sum(x**2)
    mean2 = np.sum(mean**2)
    return np.sqrt(x2 + mean2 - 2*np.dot(x, mean.T))

def euclidean_distance_matrix(test_projection, train_projection, train_person_label):
    person_mean = get_person_mean(train_projection, train_person_label)
    distance = []
    for i in range(len(test_projection)):
        distance.append([])
        for j in range(len(person_mean)):
            distance[i].append(euclidean_distance(test_projection[i], person_mean[j]))
    return np.array(distance)

def get_threshold(distance, threshold):
    return np.array([i for i in distance if i < threshold])

def get_accuracy(threshold, len_a):
    return len(threshold) / len_a


def main():
    # 0~40 labels person, each 10 images
    images, labels = load_images('att_faces/*.png')
    # split train and test data
    train_image, train_label, test_image, test_label = split_data(images, labels)
    train_person_label = train_label // 10
    test_person_label = test_label // 10
    # get pca train data
    flaten_train_image = train_image.flatten().reshape(len(train_image), -1).astype(np.float32)
    mean, cov, eigenvalue, eigenvector = get_pca(flaten_train_image)

    # visualize eigenvector
    eigenvector_gird = []
    for i in range(10):
        eigenvector_img = eigenvector[:, i].reshape(56, 46)
        eigenvector_img = (eigenvector_img - np.min(eigenvector_img)) / (np.max(eigenvector_img) - np.min(eigenvector_img))
        eigenvector_img = (eigenvector_img * 255).astype(np.uint8)
        eigenvector_gird.append(eigenvector_img)
    # make grid image of eigenvector
    eigenvector_gird = np.array(eigenvector_gird)
    eigenvector_gird = eigenvector_gird.reshape(1, 10, 56, 46)
    eigenvector_gird = np.vstack([np.hstack(i) for i in eigenvector_gird])
    cv2.imshow('eigenvector', eigenvector_gird)
    cv2.imwrite('eigenvector.png', eigenvector_gird)

    # visualize reconstruct_face
    reconstruct_face_grid = []
    for i in range(1,360,9):
        for j in [1, 10, 100, 200]:
            projection_face = projection_on_eigenvector(train_image[i].flatten(), mean, eigenvector, top_k=j)
            reconstruct_face = np.dot(projection_face, eigenvector[:, :j].T) + mean
            reconstruct_face = reconstruct_face.reshape(56, 46).astype(np.uint8)
            reconstruct_face_grid.append(reconstruct_face)
    reconstruct_face_grid = np.array(reconstruct_face_grid)
    reconstruct_face_grid = reconstruct_face_grid.reshape(8, 20, 56, 46)
    reconstruct_face_grid = np.vstack([np.hstack(i) for i in reconstruct_face_grid])
    cv2.imshow('eigenface', reconstruct_face_grid)
    cv2.imwrite('reconstruct_face.png', reconstruct_face_grid)
    cv2.waitKey(0)

    # get projection
    train_projection = get_projection(train_image, mean, eigenvector, top_k=200)
    test_projection = get_projection(test_image, mean, eigenvector, top_k=200)
    print(f'train_projection shape: {train_projection.shape}')
    print(f'test_projection shape: {test_projection.shape}')

    # # visualize train_projection using t-SNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_data = train_projection.real.astype(np.float32)
    train_tsne = tsne.fit_transform(tsne_data)
    import matplotlib.pyplot as plt
    plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=train_person_label)
    plt.show()

    # make distance matrix
    # distance_matrix = mahalanobis_distance_matrix(test_projection, train_projection, train_person_label)
    distance_matrix = euclidean_distance_matrix(test_projection, train_projection, train_person_label)
    # get predict label
    predict_label = np.argmin(distance_matrix, axis=1) + 1
    print(f'predict_label: {predict_label}')
    # get accuracy
    accuracy = np.sum(predict_label == test_person_label) / len(test_person_label)
    print(f'accuracy: {accuracy}')

    hobeom_image = cv2.imread('hobeom.jpeg', cv2.IMREAD_GRAYSCALE)
    hobeom_image = cv2.resize(hobeom_image, (46, 56))
    hobeom_image = hobeom_image.flatten().reshape(1,-1).astype(np.float32)
    hobeom_projection = projection_on_eigenvector(hobeom_image, mean, eigenvector, top_k=200)
    hobeom_matrix = euclidean_distance_matrix(hobeom_projection, train_projection, train_person_label)
    print(f'hobeom_matrix: {hobeom_matrix}')
    hobeom_label = np.argmin(hobeom_matrix, axis=1) + 1
    print(f'hobeom_label: {hobeom_label}')
    hobeom_rank = np.argsort(hobeom_matrix, axis=1) + 1
    print(f'hobeom_rank: {hobeom_rank}')


if __name__ == '__main__':
    main()