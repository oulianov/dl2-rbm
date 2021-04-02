import matplotlib.pyplot as plt
import numpy as np
from RBM import RBM
from tools import *


class DBN:
    def __init__(self, layer_sizes):
        """On va traiter le DBN comme si c'etait une liste des RBM
        Paramètres: nb_couches doit être int > 0
        """
        self.nb_couches = len(layer_sizes) - 1
        # Un DBN est un empilement de RBM
        self.model = []
        for i in range(len(layer_sizes) - 1):
            self.model.append(RBM(layer_sizes[i], layer_sizes[i + 1]))

    def entree_sortie(self, data):
        for rbm in self.model:
            data = rbm.entree_sortie(data)
        return data

    def sortie_entree(self, data):
        for rbm in reversed(self.model):
            data = rbm.sortie_entree(data)
        return data

    def train(self, data, epochs=10, batch_size=4, learning_rate=0.1):
        """Cette fonction entraîne le Deep Belief Network.
        Paramètres:
        n_epochs:
        batch_size:
        visible_v:
        lr:
        """
        x = data
        for i in range(self.nb_couches):
            print(f"Layer {i}")
            self.model[i].train(x, epochs, batch_size, learning_rate, verbose=False)
            x = self.model[i].entree_sortie(x)
        h = self.entree_sortie(data)
        data_recons = self.sortie_entree(h)
        recc_err = np.sum((data - data_recons) ** 2) / data.shape[0]
        print(f"DBN final reconstruction error: {recc_err:.2f}")
        return self

    def generer_image(self, nb_images, iter_gibs, im_shape, display=True):
        generated_images = np.empty([nb_images] + list(im_shape))
        for i in range(nb_images):
            # Gibs sampling sur le dernier layer
            v = self.model[-1].gibs_sampling(iter_gibs)
            # Propage le résultat dans les layers précédents
            # for layer in range(1, self.nb_couches):
            #     v = self.model[-(layer + 1)].sortie_entree(v)
            v = self.sortie_entree(v)
            # Binariser
            v = v > 0.5
            # Reshape
            img = v.reshape(im_shape)
            generated_images[i, :] = img.copy()
            if display:  # Display the generated image
                plt.imshow(img)
                plt.colorbar()
                plt.show()
        return generated_images


if __name__ == "__main__":
    X, im_shape = lire_alpha_digits("A")
    # Montre les 2 premiers samples
    for i in range(2):
        plt.imshow(X[i, :].reshape(im_shape))
        plt.colorbar()
        plt.show()

    dbn = DBN([X.shape[1], 64, 64, 64])
    dbn.train(X, epochs=100, batch_size=32, learning_rate=0.1)
    _ = dbn.generer_image(4, 40, im_shape)
