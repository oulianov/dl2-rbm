#%%
import matplotlib.pyplot as plt
import numpy as np 
from RBM import RBM
from lire_alpha_digits import lire_alpha_digits
from sample_bernoulli import sample_bernoulli

class DBN():
    """Create a Deep Belief Network"""
    def __init__(self, nb_couches, visible_dim, hidden_dim):
        '''On va traiter le DBN comme si c'etait une liste des RBM
        Paramètres: nb_couches doit être int > 0
        '''
        try:
            assert int(nb_couches) > 0
        except Exception:
            print("S'il vous plaît, rentrez le nombre de couche comme un entier positif.")
        self.model = [RBM(visible_dim, hidden_dim) if i==0 else RBM(hidden_dim, hidden_dim) for i in range(int(nb_couches))]
        
        self.nb_couches = nb_couches


    def pretrain_DNN(self, data, epoch=10, batch_size=32, learning_rate=0.01):
        '''Cette fonction vise entraîner la Deep Belief Network.
        Paramètres:
        n_epochs:
        batch_size:
        visible_v:
        lr:
        '''
        x = data
        for i in range(self.nb_couches):

            self.model[i].train(x,epoch, batch_size, learning_rate)
            x = self.model[i].entree_sortie(x)

        return self

    def generer_image_DBN(self, iter_gibs, nb_images, im_shape, display=True):
        """Just applied Nicola's code to a multi layer case"""
        generated_images = np.empty([nb_images] + list(im_shape))
        for i in range(nb_images):
            # Initialisation aléatoire
            v = sample_bernoulli(0.5 * np.ones(self.model[-1].visible_dim))
            #print("Shape of v",v.shape)
            for _ in range(iter_gibs):
                p_h = self.model[-1].entree_sortie(v)
                h = sample_bernoulli(p_h)
                p_v = self.model[-1].sortie_entree(h)
                v = sample_bernoulli(p_v)
            for layer in range(1, self.nb_couches):
                # print("layer: ",layer)
                v = self.model[-(layer+1)].sortie_entree(v)
                #v = sample_bernoulli(p_v)
                            
            # Reshape
            #v = self.model[0].sortie_entree(v)
            #v = sample_bernoulli(v)
            img = v.reshape(im_shape)
            generated_images[i, :] = img.copy()
            if display:  # Display the generated image
                plt.imshow(img)
                plt.show()
        return generated_images


#%%
X, im_shape = lire_alpha_digits("B")
# Montre les 2 premiers samples
for i in range(2):
    plt.imshow(X[i, :].reshape(im_shape))
    plt.show()

dbn = DBN(2,X.shape[1], 64)
dbn.pretrain_DNN(X, epoch=100, learning_rate=0.05)

dbn.generer_image_DBN(20, 4, im_shape)

# %%
