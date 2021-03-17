import numpy as np
import matplotlib.pyplot as plt

from tools import *


class RBM(object):
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

    def train(self, data, epoch=10, batch_size=32, learning_rate=0.01):
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
                proba_h_sachant_v = self.entree_sortie(data_batch)
                positive_hidden_samp = sample_bernoulli(proba_h_sachant_v)
                positive_grad = -np.transpose(data_batch) @ proba_h_sachant_v

                # Negative phase start
                hidden_samp = positive_hidden_samp

                visible_proba = self.sortie_entree(hidden_samp)
                visible_samp = sample_bernoulli(visible_proba)
                hidden_proba = self.entree_sortie(visible_samp)
                hidden_samp = sample_bernoulli(hidden_proba)

                negative_visible_samp = visible_samp
                negative_hidden_samp = hidden_samp

                negative_grad = (
                    np.transpose(negative_visible_samp) @ negative_hidden_samp
                )

                # replace reductions by lr
                grad_w_new = negative_grad - positive_grad
                grad_visible_bias_new = np.sum(
                    (data_batch - negative_visible_samp), axis=0
                )
                grad_hidden_bias_new = np.sum(
                    (proba_h_sachant_v - negative_hidden_samp), axis=0
                )
                # Update weights
                self.W += (learning_rate / real_batch_size) * grad_w_new
                self.a_bias += (learning_rate / real_batch_size) * grad_visible_bias_new
                self.b_bias += (learning_rate / real_batch_size) * grad_hidden_bias_new

            h = self.entree_sortie(data)
            data_recons = self.sortie_entree(h)

            recc_err = np.sum((data - data_recons) ** 2)

            print(f"Epoch: {i+1}/{epoch}. Reconstruction error: {recc_err}")
        return self

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
            # Initialisation aléatoire
            v = sample_bernoulli(0.5 * np.ones(self.visible_dim))
            # Gibs sampling
            for _ in range(iter_gibs):
                p_h = self.entree_sortie(v)
                h = sample_bernoulli(p_h)
                p_v = self.sortie_entree(h)
                v = sample_bernoulli(p_v)
            # Reshape
            img = v.reshape(im_shape)
            generated_images[i, :] = img.copy()
            if display:  # Display the generated image
                plt.imshow(img)
                plt.show()
        return generated_images


def main():
    X, im_shape = lire_alpha_digits("Z")
    # Montre les 2 premiers samples
    for i in range(2):
        plt.imshow(X[i, :].reshape(im_shape))
        plt.show()

    rbm = RBM(X.shape[1], 64)
    rbm.train(X, epoch=200, learning_rate=0.05)
    rbm.generer_image(4, 20, im_shape)


if __name__ == "__main__":
    main()