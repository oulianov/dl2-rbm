import matplotlib.pyplot as plt
import numpy as np
from RBM import RBM
from tools import *


class DBN:
    def __init__(self, nb_couches, visible_dim, hidden_dim):
        """On va traiter le DBN comme si c'etait une liste des RBM
        Paramètres: nb_couches doit être int > 0
        """
        try:
            assert int(nb_couches) > 0
        except Exception:
            print("Le nombre de couche doit être un entier positif.")

        self.nb_couches = nb_couches
        self.model = [RBM(visible_dim, hidden_dim)]
        for i in range(nb_couches - 1):
            self.model.append(RBM(hidden_dim, hidden_dim))
    
    def entree_sortie(self, data):
        for rbm in self.model:
            data = rbm.entree_sortie(data)
        return data
    
    def sortie_entree(self, data):
        for rbm in reversed(self.model):
            data = rbm.sortie_entree(data)
        return data

    def train(self, data, epochs=10, batch_size=4, learning_rate=0.01):
        """Cette fonction entraîne le Deep Belief Network.
        Paramètres:
        n_epochs:
        batch_size:
        visible_v:
        lr:
        """
        
        for k in range(epochs):
            x = data
            for i in range(self.nb_couches):
                self.model[i].train(x, 1, batch_size, learning_rate, verbose=False)
                x = self.model[i].entree_sortie(x)
            h = self.entree_sortie(data)
            data_recons = self.sortie_entree(h)
            recc_err = np.sum((data - data_recons) ** 2)
            print(f"DBN Epoch: {k+1}/{epochs}. Reconstruction error: {recc_err}")
        return self

    def generer_image(self, nb_images, iter_gibs, im_shape, display=True):
        generated_images = np.empty([nb_images] + list(im_shape))
        for i in range(nb_images):
            # Gibs sampling sur le dernier layer
            v = self.model[-1].gibs_sampling(iter_gibs)
            # Propage le résultat dans les layers précédents
            for layer in range(1, self.nb_couches):
                v = self.model[-(layer + 1)].sortie_entree(v)
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
    X, im_shape = lire_alpha_digits("B")
    # Montre les 2 premiers samples
    for i in range(2):
        plt.imshow(X[i, :].reshape(im_shape))
        plt.colorbar()
        plt.show()

    dbn = DBN(3, X.shape[1], 64)
    dbn.train(X, epoch=100, batch_size=32, learning_rate=0.1)
    _ = dbn.generer_image(4, 40, im_shape)
