import numpy as np

from RBM import RBM
from DBN import DBN
from tools import *


<<<<<<< HEAD
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
    m = x_ant.shape[0]
    return x_ant.T @ (x_post - y)/m


=======
>>>>>>> 08bdb9e1f36228f7dc54c224d8de99b2a1bd3fcf
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
        self.logits = x @ self.W + self.b
        self.y = self.activation(self.logits)
        return self.y

    def backward(self, true_y, previous_y, next_layer, lr):
        # d L / d x pour une cross-entropy loss
        if next_layer is None:
            c = self.y - true_y
        else:
            c = next_layer.d_logits * self.d_activation(self.y)
        # d L / d W
        d_W = previous_y.T @ c
        # d L / d b
        d_b = c.sum(axis=0)
        # Mémoriser le gradient de la sortie de la couche avant activation
        self.d_logits = c @ self.W.T
        # Gradient descent
        batch_size = c.shape[0]
        self.W -= (lr / batch_size) * d_W
        self.b -= (lr / batch_size) * d_b

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
    def __init__(self, layer_sizes):
        # Input Layer
        self.layers = []
        # Middle dimensions
        for i in range(len(layer_sizes) - 1):
            if i + 1 < len(layer_sizes) - 1:
                # Middle layer : relu activation
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
            else:
                # Output layer : sigmoid activation
                self.layers.append(
                    Layer(layer_sizes[i], layer_sizes[i + 1], softmax, d_softmax)
                )

    def init_DNN_with_DBN(self, DBN):
        for i, rbm in DBN.model:
            self.layers[i].init_Layer_to_RBM(rbm)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def entree_sortie_reseau(self, data: np.array) -> list:
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
        valeur_layer = []
        y = data
        for layer in dnn.layers:
            y = layer.forward(y)
            valeur_layer.append(y)
        # La dernière valeur est la proba
        return valeur_layer

    def backward(self, valeur_layer, true_y, lr):
        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            estimated_y = valeur_layer[i]
            previous_y = valeur_layer[i - 1]
            if i < len(self.layers) - 1:
                next_layer = self.layers[i + 1]
            else:
                next_layer = None
            previous_y = layer.backward(true_y, previous_y, next_layer, lr)

    def train(self, x, y, epochs=10, learning_rate=0.1, batch_size=128):
        for i in range(epochs):
            # Shuffle data
            random_index = np.random.choice(x.shape[0], x.shape[0], replace=False)
            x_shuffled = x[random_index, :]
            y_shuffled = y[random_index, :]
            # Perform gradient descent for each batch
            for j in range(0, x.shape[0], batch_size):
                # Select data for batch
                max_index = min(batch_size + j, x_shuffled.shape[0] - 1)
                x_batch = x_shuffled[j:max_index]
                y_batch = y_shuffled[j:max_index]
                # Forward pass
                valeur_layer = self.entree_sortie_reseau(x_batch)
                # Backward pass
                self.backward(valeur_layer, y_batch, learning_rate)
            # Print progress
            print(f"Epoch {i+1} : Loss {self.loss(self.forward(x), y)}")

    def loss(self, y, true_y):
        loss = true_y * np.log(y) + (1 - true_y) * np.log(1 - y)
        return -loss.mean()

    def fine_tuning(self, x, y):
        pass

    def predict(self, x):
        return (self.forward(x) > 0.5).astype(int)


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


def one_hot_encode(y, nb_classes):
    return np.eye(nb_classes)[y]


def test_dnn(dnn, X_test, y_test):
    """Returns error rate"""
    y_predict = dnn.predict(X_test)
    accuracy = np.mean(y_test == y_predict)
    error_rate = 1 - accuracy
    return error_rate


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = lire_mnist()
    y_train = one_hot_encode(y_train, 10)
    y_test = one_hot_encode(y_test, 10)
    # Flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    dnn = DNN([784, 128, 10])
    dnn.train(X_train, y_train, learning_rate=0.5, epochs=10)
    print(f"Error rate: {test_dnn(dnn, X_test, y_test)}")