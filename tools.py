import numpy as np
import string
import os
import scipy.io
import string
import imageio
import idx2numpy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    z = sigmoid(x)
    return z * (1 - z)


def softmax(x):
    z = np.exp(x - np.max(x, axis=1, keepdims=True))
    return z / np.sum(z, axis=1, keepdims=True)


def d_softmax(x):
    z = softmax(x)
    return -z.T @ z + np.diagflat(z)


def relu(x):
    return x * (x > 0)


def d_relu(x):
    return (x > 0) * 1  # to return int


def sample_bernoulli(proba):
    """Prend un échantillon d'une distribution de Bernoulli
    Arguments:
        proba: Distribution de proba de laquelle prendre un échantillon
    Return:
        sample: échantillon de la distribution
    """
    sample = (np.random.random(proba.shape) < proba) * 1
    return sample


def lire_alpha_digits(caractere):
    """Permet de récupérer les données sous forme matricielle
    ----------
    caractere : donner le caractère à apprendre, peut être un
    entier entre 0 et 9 ou bien une lettre de l'alphabet en majuscule
    ou bien leur indice entre 0 et 35
    -------
    Une matrice dont les lignes représentes les données et les colonnes les composantes
    propre au caractère demandé
    """
    cwd = os.getcwd()
    mat = scipy.io.loadmat(cwd + "/data/binaryalphadigs.mat")

    if type(caractere) == int and caractere < 36 and caractere >= 0:
        data = mat.get("dat")[caractere]
    elif type(caractere) == str and len(caractere) == 1:
        data = mat.get("dat")[10 + string.ascii_uppercase.index(caractere)]
    else:
        raise ValueError("Data not available for that caracter")
    data = np.array([i for i in data])
    im_shape = data.shape[1:]
    data = data.reshape(data.shape[0], -1)
    return data, im_shape


def lire_mnist():
    """Lire les données MNIST.

    Exemple:
    X_train, y_train, X_test, y_test = lire_mnist()

    Returns:
        images_train (np.array):
        label_train (np.array):
        images_test (np.array):
        label_test (np.array):
    """
    cwd = os.getcwd()
    data_directory = cwd + "/data/mnist/"
    files = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    output = []
    for file in files:
        output.append(idx2numpy.convert_from_file(data_directory + file))
    return tuple(output)


def save_merged_images(images, size, path):
    """This function concatenate multiple images and saves them as a single image.
    Args:
        images: images to concatenate
        size: number of columns and rows of images to be concatenated
        path: location to save merged image
    Returns:
        saves merged image in path
    """
    h, w = images.shape[1], images.shape[2]

    merge_img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        merge_img[j * h : j * h + h, i * w : i * w + w] = image

    imageio.imwrite(path, merge_img)