#%%
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

    def pretrain_DNN(self, data, epoch=10, batch_size=4, learning_rate=0.01):
        """Cette fonction vise entraîner la Deep Belief Network.
        Paramètres:
        n_epochs:
        batch_size:
        visible_v:
        lr:
        """
        x = data
        for i in range(self.nb_couches):
            self.model[i].train(x, epoch, batch_size, learning_rate)
            x = self.model[i].entree_sortie(x)

        return self

    def generer_image(self, nb_images, iter_gibs, im_shape, display=True):
        """Just applied Nicola's code to a multi layer case"""
        generated_images = np.empty([nb_images] + list(im_shape))
        for i in range(nb_images):
            # Initialisation aléatoire
            img = np.zeros(self.model[0].visible_dim)
            for j in range(10):
                v = sample_bernoulli(0.5 * np.ones(self.model[-1].visible_dim))
                for k in range(iter_gibs):
                    p_h = self.model[-1].entree_sortie(v)
                    h = sample_bernoulli(p_h)
                    p_v = self.model[-1].sortie_entree(h)
                    v = sample_bernoulli(p_v)
                transf = v
                for layer in range(1, self.nb_couches):
                    transf = self.model[-(layer + 1)].sortie_entree(transf)
                    transf = sample_bernoulli(transf)
                img += transf
            # Magic sampling ?
            img = sample_bernoulli(img)
            img = img.reshape(im_shape)
            # Magic rescale ?
            img = 1 - img
            generated_images[i, :] = img.copy()
            if display:  # Display the generated image
                plt.imshow(img)
                plt.colorbar()
                plt.show()
        return generated_images


#%%
X, im_shape = lire_alpha_digits("B")
# Montre les 2 premiers samples
for i in range(2):
    plt.imshow(X[i, :].reshape(im_shape))
    plt.colorbar()
    plt.show()

#%%
dbn = DBN(3, X.shape[1], 30)
dbn.pretrain_DNN(X, epoch=20, batch_size=5, learning_rate=0.1)
_ = dbn.generer_image(4, 20, im_shape)

# %%
