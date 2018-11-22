import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts

def plotFigures(figures, size=5):
    """
    :param figures: List of figures to plot
    :param title: Title to give to the figures
    :param size: Width size of the grid
    """

    for img_set, images in figures.items():
        plt.figure(figsize=(15, 15))
        for i, img in enumerate(images, 1):
            plt.subplot(size, size, i), plt.imshow(img, 'gray')
            plt.title(f'{img_set} - {i}'), plt.axis('off')
        plt.show()


test_path = 'dataset/test/group.jpg'
train_path = 'dataset/train/'

# Detect all faces in group picture
face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
group = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
group_faces = face_cascade.detectMultiScale(group, 1.2, 4)

# Draw a box around each detected face
group_copy = group.copy()
for (x, y, w, h) in group_faces:
    cv2.rectangle(group_copy, (x, y), (x+w, y+h), (125, 125, 125), 5)

# Display test image
plt.figure(figsize=(30, 15))
plt.title("Group Picture with Detected Faces")
plt.imshow(group_copy, 'gray'), plt.axis('off')
plt.show()

# Read all training images
X, Y = [], []
train_sets = {}
size = (128, 128)
for img_set in os.listdir(train_path):
    images = []
    for img in os.listdir(os.path.join(train_path, img_set)):
        face = cv2.imread(os.path.join(train_path, img_set, img), cv2.IMREAD_GRAYSCALE)
        p = face_cascade.detectMultiScale(face, 1.05, 15)
        if len(p):
            x, y, w, h = p[0]
            face = face[y:y+h, x:x+w]
        face = cv2.resize(face, size)
        images.append(face)
        X.append(face), Y.append(img_set)
    train_sets[img_set] = images

# Display training images
plotFigures(train_sets)


##################################################

# Split the data using train_test_split
X_train, X_test, Y_train, Y_test = tts(X, Y, train_size=2/3)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Using the object, transform the test set
X_test_pca = pca.transform(X_test)

# Print the shape of the transformed data
print(X_train_pca.shape)

# Get the variance explained by every principal component
print(pca.explained_variance_ratio_)

for c in train_sets.keys():
    x_cls = X_train_pca[Y_train == c, :]
    plt.scatter(x_cls[:,0], x_cls[:,1])

plt.show()
