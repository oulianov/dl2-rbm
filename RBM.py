import scipy.io
import os
import string
import numpy as np
import tensorflow as tf

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
    mat = scipy.io.loadmat(cwd + '/data/binaryalphadigs.mat')

    if type(caractere) == int and caractere < 36 and caractere >= 0:
        return mat.get('dat')[caractere]
    elif type(caractere) == str and len(caractere) == 1:
        return mat.get('dat')[10 + string.ascii_uppercase.index(caractere)]
    else : raise ValueError("Data not available for that caracter")

def sample_bernoulli(proba):
    """ Prend un échantillon d'une distribution de Bernoulli
    Arguments:
        proba: Distribution de proba de laquelle prendre un échantillon
    Return:
        sample: échantillon de la distribution
    """
    sample = tf.nn.relu(tf.sign(proba - tf.random_uniform(tf.shape(proba))))
    return sample

#sigmoid function
def sigmoid(x):
   return 1/(1+np.exp(-x))

class RBM(object):
    def __init__(self, visible_dim, hidden_dim):
        
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim


        #create wieghts
         self.W = tf.Variable(tf.truncated_normal(
             [self.n_visible, self.n_hidden], mean=0.0, stddev=0.05, dtype=tf.float32), name="weight_matrix")

        #create bias    
        self.a_bias = tf.zeros(visible_dim)
        self.b_bias = tf.zeros(hidden_dim)

        # create input placeholders
        self.visible_vect = tf.placeholder(tf.float32, [None, self.n_visible], name="visible_input_placeholder")
        self.hidden_vect = tf.placeholder(tf.float32, [None, self.n_hidden], name="hidden_input_placeholder")

        

    def _entree_sortie_(self, visible_v):
        """Retourne la valeur des unités de sortie calculées à partir de la fonction sigmoïde
        ----------
        visible_v : vecteur des données d'entrée
        -------
        Return : vecteur des unités de sortie
        """
        return tf.nn.sigmoid(tf.matmul(visible_v, self.W) + b_bias)

    
    def _sortie_entree_(self, hidden_v):
        """Retourne la valeur des unités de sortie calculées à partir de la fonction sigmoïde
        ----------
        hidden_v : vecteur des données de sortie
        -------
        Return : vecteur des unités d'entrée '
        """
        return tf.nn.sigmoid(tf.matmul(hidden_v, tf.transpose(self.W)) + a_bias)

    def _contrastive_divergence(self):
        input = self.visible_vect



        #Positive phase
        proba_h_sachant_v = self._entree_sortie_(input)
        positive_hidden_samp = sample_bernoulli(proba_h_sachant_v)
        positive_grad = - tf.matmul(proba_h_sachant_v,tf.transpose(input))

        #Negative phase start
        hidden_samp = positive_hidden_samp
        
        visible_proba = self._sortie_entree_(hidden_samp)
        visible_samp = sample_bernoulli(visible_proba)
        hidden_proba = self._entree_sortie_(visible_samp)
        hidden_samp = sample_bernoulli(hidden_proba)

        negative_visible_samp = visible_samp
        negative_hidden_samp = hidden_samp
        
        negative_grad = tf.matmul(tf.transpose(negative_visible_samp), negative_hidden_samp)

        grad_w_new = -(positive_grad - negative_grad) / tf.to_float(tf.shape(input)[0])
        grad_visible_bias_new = -(tf.reduce_mean(input - negative_visible_samp, 0))
        grad_hidden_bias_new = -(tf.reduce_mean(proba_h_sachant_v - negative_hidden_samp, 0))

        grads = [grad_w_new, grad_visible_bias_new, grad_hidden_bias_new]

        return grads
    



    


