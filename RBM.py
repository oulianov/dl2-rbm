import numpy as np
import matplotlib.pyplot as plt

from tools import *


class RBM:
    def __init__(self, visible_dim, hidden_dim):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        # create weights
        self.W = np.random.normal(
            loc=0.0, scale=0.01, size=(self.visible_dim, self.hidden_dim)
        )
        # Bias
        self.a_bias = np.zeros(visible_dim)
        self.b_bias = np.zeros(hidden_dim)

        # create input placeholders
        self.visible_vect = np.empty((self.visible_dim), dtype=np.float32)
        self.hidden_vect = np.empty((self.hidden_dim), dtype=np.float32)

    def entree_sortie(self, visible_v):
        """Retourne la valeur des unités de sortie calculées à partir de la fonction sigmoïde
        ----------
        visible_v : vecteur des données d'entrée
        -------
        Return : vecteur des unités de sortie
        """
        return sigmoid(visible_v @ self.W + self.b_bias)
        # (39,320)

    def sortie_entree(self, hidden_v):
        """Retourne la valeur des unités de sortie calculées à partir de la fonction sigmoïde
        ----------
        hidden_v : vecteur des données de sortie
        -------
        Return : vecteur des unités d'entrée '
        """
        return sigmoid(hidden_v @ np.transpose(self.W) + self.a_bias)

    def train(self, data, epoch=10, batch_size=32, learning_rate=0.1, verbose=True):
        """Entraîne le RBM avec l'algorithme de contrastive divergence.
        ------
        data : (np.array) données d'entraînement, de dimension (n_samples, self.visible_dim)
        epoch : (int) nombre d'époques d'entraînement
        batch size : (int) taille du batch
        ------
        Return : self (RBM entraîné)
        """
        for i in range(epoch):
            # Shuffle the data
            data_shuffled = data[
                np.random.choice(data.shape[0], data.shape[0], replace=False), :
            ]
            for j in range(0, data.shape[0], batch_size):
                data_batch = data_shuffled[
                    j : min(batch_size + j, data_shuffled.shape[0] - 1)
                ]
                real_batch_size = data_batch.shape[0]  # might be lower than batch_size

                # Positive phase
                v_0 = data_batch
                p_h_v_0 = self.entree_sortie(v_0)
                h_0 = sample_bernoulli(p_h_v_0)

                # Negative phase start
                p_v_h_0 = self.sortie_entree(h_0)
                v_1 = sample_bernoulli(p_v_h_0)
                p_h_v_1 = self.entree_sortie(v_1)

                # Gradient
                d_a = np.sum(v_0 - v_1, axis=0)
                d_b = np.sum(p_h_v_0 - p_h_v_1, axis=0)
                d_W = v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1

                # Mise à jour des poids
                self.W += (learning_rate / real_batch_size) * d_W
                self.a_bias += (learning_rate / real_batch_size) * d_a
                self.b_bias += (learning_rate / real_batch_size) * d_b

            # Calcul de l'erreur de reconstruction
            h = self.entree_sortie(data)
            data_recons = self.sortie_entree(h)
            recc_err = np.sum((data - data_recons) ** 2) / data.shape[0]
            if verbose == True:
                print(f"Epoch: {i+1}/{epoch}. Reconstruction error: {recc_err:.2f}")
        return self

    def gibs_sampling(self, iter_gibs=20, init=False):
        if init is False:
            # Initialisation aléatoire
            v = sample_bernoulli(0.5 * np.ones(self.visible_dim))
        else:
            # Initialisation avec un vecteur donné
            v = init
        for _ in range(iter_gibs):
            p_h = self.entree_sortie(v)
            h = sample_bernoulli(p_h)
            p_v = self.sortie_entree(h)
            v = sample_bernoulli(p_v)
        return v

    def generer_image(self, nb_images, iter_gibs, im_shape, display=True):
        """Génère des images grâce au RBM.

        Args:
            nb_images (int): quantité d'images à générer
            iter_gibs (int): nombre d'itération à utiliser pour le Gibs
                sampling
            im_shape (list-like) : taille de l'image générée (pour le reshape)
            display (bool) : montrer les images ou non
        Returns:
            generated_images (np.array) : les images générées, un array de
                dimension (nb_images, im_shape)
        """
        generated_images = np.empty([nb_images] + list(im_shape))
        for i in range(nb_images):
            # Gibs sampling
            v = self.gibs_sampling(iter_gibs)
            # Reshape
            img = v.reshape(im_shape)
            generated_images[i, :] = img.copy()
            if display:  # Display the generated image
                plt.imshow(img)
                plt.show()
        return generated_images


if __name__ == "__main__":
    X, im_shape = lire_alpha_digits("F")
    # Montre les 2 premiers samples
    for i in range(2):
        plt.imshow(X[i, :].reshape(im_shape))
        plt.show()

    rbm = RBM(X.shape[1], 100)
    rbm.train(X, epoch=200, learning_rate=0.05)
    rbm.generer_image(4, 20, im_shape)
