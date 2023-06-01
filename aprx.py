import random
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# Load faces dataset
faces = fetch_olivetti_faces()
X = faces.data
n_samples, n_features = X.shape

sum_img = np.zeros((faces.images.shape[1], faces.images.shape[2]))
new_X = np.empty([400, faces.images.shape[1]*faces.images.shape[2]])
mean_img = np.zeros([faces.images.shape[1], faces.images.shape[2]])

for val in X:
    val = val.reshape(faces.images.shape[1], faces.images.shape[2])
    sum_img =sum_img+val

a = len(X)
mean_img = sum_img/len(X)
for i in range(len(X)):
    X_reshape = X[i].reshape(faces.images.shape[1], faces.images.shape[2])
    mean = np.mean(X_reshape, axis=0)
    new_X[i] = (X_reshape - mean_img).flatten()

cov_matrix = np.dot(new_X.T, new_X)
n_components = 3662  # Number of principal components to keep

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
eigenvectors = eigenvectors[:,:n_components]
Y = np.dot(eigenvectors.T,new_X.T)
Y=Y.T
eigenvectors = eigenvectors.T
# Plot original and approximated faces
plt.ion()
fig1, (ax_2) = plt.subplots(1, 1, figsize=(10, 4), subplot_kw={'xticks': [], 'yticks': []})
futpic = 0
for i in range(10,60,10):
    # Approximate faces with principal components
    sample_index = i  # Index of sample to approximate
    sample_face = new_X[sample_index].reshape(faces.images.shape[1], faces.images.shape[2])
    Y_i = Y[sample_index]
    Y_0 = Y[sample_index - 10]
    k = 0.2
    h =[0]*n_components
    for p in range(0,50,1):
        for u in range(len(Y_i)):
            h[u] += Y_0[u].real
        for u in range(len(Y_i)):
            if h[u] < 0:
                g = abs(Y_0[u])
                Y_0[u] += (abs(Y_0[u]) + Y_0[u] + Y_i[u])*k
            else:
                Y_0[u] += (Y_0[u] - Y_0[u] + Y_i[u])*k

        aprx_img = np.zeros((faces.images.shape[1], faces.images.shape[2]))
        g = len(eigenvectors)
        for i in range(len(Y_0)):
            aprx_img =aprx_img + Y_0[i]*(eigenvectors[i].reshape(faces.images.shape[1], faces.images.shape[2]))
        aprx_img += mean_img
        aprx_img = aprx_img.real
        sample_face =faces.images[sample_index]
        # Calculate percentage of convergence
        mse = np.mean((sample_face - aprx_img) ** 2)
        # Reconstruct image using principal components
        ax_2.imshow(aprx_img, cmap='gray')
        fig1.canvas.draw()
        fig1.canvas.flush_events()
    ax_2.set_title('Нажми на лицо')
    plt.waitforbuttonpress()
    ax_2.set_title('')
    futpic+=1

plt.waitforbuttonpress()
