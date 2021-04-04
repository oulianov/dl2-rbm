import matplotlib.pyplot as plt

from DBN import *
from DNN import *
from tools import *


def plot_results(error_random_all, error_pretrained_all, horizontal, x_label, title):
    plt.plot(horizontal, error_random_all, label="Random init")
    plt.plot(horizontal, error_pretrained_all, label="DBN Pretrain")
    plt.ylabel("Error rate")
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}.png", dpi=144.0, transparent=False)
    plt.show()


def compare_dnn_init(X_train, y_train, X_test, y_test, layers=[784, 100, 10]):
    print(f"Architecture : {layers}")
    # 1. initialiser deux réseaux identiques;
    dnn_random = DNN(layers)
    dnn_pretrained = DNN(layers)

    # 2. pré-apprendre un des deux réseau en le considérant comme un empilement de RBM (apprentissage non
    # supervisé);
    layers_dbn = layers[:-1]
    dbn = DBN(layers_dbn)
    print("Pretraining RBM...")
    dbn.train(X_train, epochs=100, batch_size=32, learning_rate=0.1)
    dnn_pretrained.init_DNN_with_DBN(dbn)

    # 3. apprendre le réseau pré-appris préalablement avec l’algorithme de rétro-propagation;
    dnn_pretrained.train(X_train, y_train, epochs=100, batch_size=32, learning_rate=0.1)

    # 4. apprendre le second réseau qui a été initialisé aléatoirement avec l’algorithme de rétro-propagation;
    dnn_random.train(X_train, y_train, epochs=100, batch_size=32, learning_rate=0.1)

    # 5. Calculer les taux de mauvaises classifications avec le réseau 1 (pré-entrainé + entraîné) et le réseau 2
    # (entraîné) à partir du jeu ’train’ et du jeu ’test’
    error_random = dnn_random.error_rate(X_test, y_test)
    error_pretrained = dnn_pretrained.error_rate(X_test, y_test)
    print(f"Error rate random init : {error_random:.3f}")
    print(f"Error rate RBM pretrained : {error_pretrained:.3f}")

    return error_random, error_pretrained


# Lire les données
X_train, y_train, X_test, y_test = lire_mnist()
y_train = one_hot_encode(y_train, 10)
y_test = one_hot_encode(y_test, 10)
# Flatten
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
# Binarize
X_train = (X_train > 0.5) * 1
X_test = (X_test > 0.5) * 1

# L'entraînement prend beaucoup de temps, donc on le fait seulement sur un subset
X_train_small = X_train[:1000]
y_train_small = y_train[:1000]


# Fig 1 : Comparing nb layers

error_random_all = []
error_pretrained_all = []
for nb_layers in range(2, 6):
    layers = [784] + nb_layers * [200] + [10]
    error_random, error_pretrained = compare_dnn_init(
        X_train_small, y_train_small, X_test, y_test, layers
    )
    error_random_all.append(error_random)
    error_pretrained_all.append(error_pretrained)
print("Random:", error_random_all)
print("Pretrain:", error_pretrained_all)

# Random: [0.26170000000000004, 0.2479, 0.45030000000000003, 0.793]
# Pretrain: [0.12170000000000003, 0.11219999999999997, 0.13080000000000003, 0.14870000000000005]

plot_results(
    error_random_all,
    error_pretrained_all,
    list(range(2, 6)),
    "Depth (nb layers)",
    "Comparing depth",
)


# Fig 2 : Comparing width layers

layer_sizes = list(range(100, 800, 100))
error_random_all = []
error_pretrained_all = []
for layer_size in layer_sizes:
    layers = [784] + 2 * [layer_size] + [10]
    error_random, error_pretrained = compare_dnn_init(
        X_train_small, y_train_small, X_test, y_test, layers
    )
    error_random_all.append(error_random)
    error_pretrained_all.append(error_pretrained)
print("Random:", error_random_all)
print("Pretrain:", error_pretrained_all)
plot_results(
    error_random_all,
    error_pretrained_all,
    layer_sizes,
    "Layer width",
    "Comparing layer width",
)


# Fig 3 : Comparing data size

training_sizes = [1000, 3000, 7000, 10000, 30000, 60000]
layers = [784, 200, 200, 10]
error_random_all = []
error_pretrained_all = []
for training_size in training_sizes:
    sub_X = X_train[:training_size]
    sub_y = y_train[:training_size]
    error_random, error_pretrained = compare_dnn_init(
        sub_X, sub_y, X_test, y_test, layers
    )
    error_random_all.append(error_random)
    error_pretrained_all.append(error_pretrained)
print("Random:", error_random_all)
print("Pretrain:", error_pretrained_all)
plot_results(
    error_random_all,
    error_pretrained_all,
    training_sizes,
    "Training size",
    "Comparing traing sizes",
)


# On cherchera enfin une configuration permettant d’obtenir le meilleur
# taux de classification possible (ne pas hésiter à utiliser les 60000 données
# et des réseaux de grande taille).

dnn = DNN([784, 600, 400, 200, 10])
dbn = DBN([784, 600, 400, 200])
print("Pretraining RBM...")
dbn.train(X_train, epochs=200, batch_size=64, learning_rate=0.1)
dnn.init_DNN_with_DBN(dbn)
dnn.train(X_train, y_train, epochs=150, batch_size=32, learning_rate=0.1)
error = dnn.error_rate(X_test, y_test)
print(f"Meilleure erreur: {error}")