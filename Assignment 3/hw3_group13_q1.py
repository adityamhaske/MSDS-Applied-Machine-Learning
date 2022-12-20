# %%
import numpy as np
import pandas as pd

import cv2
#Used colab for programming- as cv2.imshow causes crashing the kernel, following method was used
# from google.colab.patches import cv2_imshow

import matplotlib.pyplot as plt

# %%
#reading an image into numpy array
array_im = cv2.imread("D:\IUB\COURSE WORK\FALL 2022\APPLIED MACHINE LEARNING\ASSIGNMENTS\ASSIGNMENT 3\hw3_image_1.jpg")

# %%
#### Close the image window to continue
cv2.imshow(" ",array_im)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
print(array_im.shape)

# %% [markdown]
# To use the colored image for PCA, we need to apply PCA on individual R, G, B channels of an image.
# 
# Reference tutorials for "split" method: 
# 1. https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118

# %%
#getting the r, g, b components of an image
im_r, im_g, im_b = cv2.split(array_im)

# %%
im_r.shape

# %%
#function to calculate means
def calc_mean(img):
  mean = np.mean(img, axis = 0)
  return mean

# %%
#function to get eigen vectors and eigenvalues[numpy documentation]
def eigen(mat):
  eigen_val, eigen_vec = np.linalg.eig(mat)
  return eigen_val, eigen_vec

# %% [markdown]
# References: {referred but not copied from}
# 1. [formulae, and methods] In-class module- slides [PCA for dimensionality reduction]
# 2. [for grayscale image reconstruction] https://medium.com/@pranjallk1995/pca-for-image-reconstruction-from-scratch-cf4a787c1e36

# %%
def method_pca(im, pc):
  mean = calc_mean(im)

  #subtracting mean from image array
  X_red = im - mean # X_reduced

  #getting covariance matrix
  im_cov = np.cov(X_red)

  #getting eigen_values and eigen_vectors
  e_val, e_vec = eigen(im_cov)

  #sorting eigen_values and eigen_vectors in descending order[argsort documentation]
  eig_sort = np.argsort(e_val)[::-1]

  eval_sort = e_val[eig_sort]
  evec_sort = e_vec[:, eig_sort]

  #for calculation of variance explained: 
  #referenced for understanding [but not copied]: https://vitalflux.com/pca-explained-variance-concept-python-example/#:~:text=Explained%20variance%20is%20calculated%20as,decomposition%20PCA%20class.
  eval_sum = sum(eval_sort)
  var_explained = [(i / eval_sum) for i in eval_sort]

  #here the number of principal components should be less than 432
  #selecting required number of components into vector
  vec = evec_sort[:, range(pc)]

  X_recov = np.dot(vec.T, X_red) # (X recovered = X_reduced . W)
  reconst = np.dot(vec, X_recov) + mean

  return reconst, var_explained

# %% [markdown]
# PCA for 10 components:

# %%
#number of principal components
pc = 10

# %% [markdown]
# implementing PCA for every channel

# %%
reconst_r, varr = method_pca(im_r, pc)
reconst_g, varg = method_pca(im_g, pc)
reconst_b, varb = method_pca(im_b, pc)

# %% [markdown]
# Combining R,G,B components.
# 

# %%
im2 = cv2.merge([reconst_r, reconst_g, reconst_b])

# %%
cv2.imshow(" ", im2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% [markdown]
# PCA for 100 components:
# 

# %%
#number of principal components
pc = 100

# %%
reconst_r, varr = method_pca(im_r, pc)
reconst_g, varg = method_pca(im_g, pc)
reconst_b, varb = method_pca(im_b, pc)

# %%
im2 = cv2.merge([reconst_r, reconst_g, reconst_b])

# %%
cv2.imshow(" ", im2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
#calculating cumulative explained variance
cumulative_v = np.cumsum(varr)

# %% [markdown]
# **Plot for accumulative variance vs number
# of principle components:**

# %%
plt.step(range(0,len(cumulative_v)), cumulative_v)
plt.ylabel('Cumulative variance')
plt.xlabel('Principal components')
plt.show()

# %% [markdown]
# PCA for 200 components:

# %%
#number of principal components
pc = 200

# %%
reconst_r, varr = method_pca(im_r, pc)
reconst_g, varg = method_pca(im_g, pc)
reconst_b, varb = method_pca(im_b, pc)

# %%
im2 = cv2.merge([reconst_r, reconst_g, reconst_b])

# %%
cv2.imshow(" ", im2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% [markdown]
# PCA for 300 components:

# %%
#number of principal components
pc = 300

# %%
reconst_r, varr = method_pca(im_r, pc)
reconst_g, varg = method_pca(im_g, pc)
reconst_b, varb = method_pca(im_b, pc)

# %%
im2 = cv2.merge([reconst_r, reconst_g, reconst_b])

# %%
cv2.imshow(" ",im2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% [markdown]
# PCA for 400 components:

# %%
#number of principal components
pc = 400

# %%
reconst_r, varr = method_pca(im_r, pc)
reconst_g, varg = method_pca(im_g, pc)
reconst_b, varb = method_pca(im_b, pc)

# %%
im2 = cv2.merge([reconst_r, reconst_g, reconst_b])

# %%
cv2.imshow(" ", im2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% [markdown]
# The number of principal components cannot exceed 432, as the dimensions of each [channel] image are (432, 768).

# %%



