import numpy as np

from RBM import RBM
from DBN import DBN
from tools import *


def d_sigmoid(x):
    z = sigmoid(x)
    return z * (1 - z)


def softmax(x):
    z = np.exp(x)
    return z / np.sum(z, axis=1, keepdims=True)


def d_softmax(x):
    z = softmax(x)
    return -z.T @ z + np.diagflat(z)


def d_loss_x(x, y):
    return softmax(x) - np.diagflat(y)


def d_loss_w(x_ant, x_post, y):
    pass


class Layer:
    def __init__(
        self,
        input_dim,
        output_dim,
        activation=sigmoid,
        d_activation=d_sigmoid,
    ):
        self.W = np.random.normal(loc=0, scale=0.1, size=(input_dim, output_dim))
        self.b = np.zeros(shape=(output_dim))
        self.activation = activation
        self.d_activation = d_activation

    def forward(self, x):
        return self.activation(x @ self.W + self.b)

    def backward(self, y, true_y):
        pass

    def init_Layer_to_RBM(self, rbm: RBM):
        try:
            assert self.W.shape == rbm.W.shape
            assert self.b.shape == rbm.b_bias.shape
        except Exception as e:
            print(
                f"Shapes not compatible between Layer and provided RBM.\
                self.W {self.W.shape} should be the same as rbm.W {rbm.W}\
                self.b {self.b.shape} should be the same as rbm.b_biais {rbm.b_bias}"
            )
        self.W = rbm.W
        self.b = rbm.b_bias


class DNN:
    def __init__(self, input_dim, hidden_dim, nb_layers, output_dim):
        # Input Layer
        self.layers = [Layer(input_dim, hidden_dim)]
        # Middle dimensions
        for i in range(nb_layers):
            self.layers.append(Layer(hidden_dim, hidden_dim))
        # Output Layer
        self.layers.append(Layer(hidden_dim, output_dim, softmax, d_softmax))

    def init_DNN_with_DBN(self, DBN):
        for i, rbm in DBN.model:
            self.layers[i].init_Layer_to_RBM(rbm)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def backward(self, y, true_y):
        pass

    def fine_tuning(self, X, y):
        pass


def calcul_softmax(rbm: RBM, data: np.array) -> np.array:
    """Prend en argument un RBM, des données d’entrée et qui
    retourne des probabilités sur les unités de sortie à partir de
    la fonction softmax.

    Args:
        rbm (RBM): [description]
        data (np.array): [description]

    Returns:
        np.array: Probabilités sur les unités de sortie
    """
    return softmax(RBM.entree_sortie(data))


def entree_sortie_reseau(dnn: DNN, data: np.array) -> list:
    """prend en argument un DNN, des données en entrée du
    réseau et qui retourne dans un tableau les sorties sur chaque
    couche cachées du réseau ainsi que les probabilités sur les
    unités de sortie. Cette fonction pourra utiliser les fonctions
    entree_sortie_RBM et calcul_softmax

    Args:
        dnn (DNN): [description]
        data (np.array): [description]

    Returns:
        list: valeur des sorties sur chaque couche
    """
    valeur_layer = [data]
    y = data
    for layer in dnn.layers:
        y = layer.forward(y)
        valeur_layer.append(y)
    # La dernière valeur est la proba
    return valeur_layer


if __name__ == "__main__":
    dnn = DNN(320, 8, 3, 2)
    X, im_shape = lire_alpha_digits("B")
    dnn.forward(X).shape  # Should be 39, 2
