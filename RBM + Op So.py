import scipy.io
import os
import string
import numpy as np
import tensorflow as tf
import scipy.misc
import imageio

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
    sample = tf.nn.relu(tf.sign(proba - tf.random.uniform(tf.shape(proba),dtype=tf.float32)))
    return sample

import scipy.misc
import numpy as np


def save_merged_images(images, size, path):
    """ This function concatenate multiple images and saves them as a single image.
    Args:
        images: images to concatenate
        size: number of columns and rows of images to be concatenated
        path: location to save merged image
    Returns:
        saves merged image in path
    """
    h, w = images.shape[1], images.shape[2]

    merge_img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        merge_img[j * h:j * h + h, i * w:i * w + w] = image

    imageio.imwrite(path, merge_img)


#sigmoid function
def sigmoid(x):
   return 1/(1+np.exp(-x))

class RBM(object):
    def __init__(self, visible_dim, hidden_dim, lr):

        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        #create wieghts
        self.W = tf.Variable(tf.compat.v1.truncated_normal([self.visible_dim, self.hidden_dim], mean=0.0, stddev=0.05, dtype=tf.float32), name="weight_matrix")
        
        #create bias    
        self.a_bias = tf.zeros(visible_dim)
        self.b_bias = tf.zeros(hidden_dim)

        # create input placeholders
        self.visible_vect = tf.compat.v1.placeholder(tf.float32, [None, self.visible_dim], name="visible_input_placeholder")
        self.hidden_vect = tf.compat.v1.placeholder(tf.float32, [None, self.hidden_dim], name="hidden_input_placeholder")

        # create optimizer
        self.lr = lr
        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr)

        # construct graph(s)
        self.v_marg_steps = 0.1

        self.train_op = self._training(grads=self._contrastive_divergence())
        self.reconstruction = self._reconstruct(self.visible_vect)
        self.reconstruction_error = tf.reduce_mean(tf.square(self.visible_vect - self.reconstruction))
        self.inferred_hidden_activations = sample_bernoulli(self._entree_sortie_(self.visible_vect))
        self.v_marg = self._gibbs_sample_v_prime_given_v(self.visible_vect, steps=self.v_marg_steps)

        self._idx_pll = 0  # index used for pll calculation
        self.pll = self._pseudo_log_likelihood()

        # init variables
        init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(init)

        

    def _entree_sortie_(self, visible_v):
        """Retourne la valeur des unités de sortie calculées à partir de la fonction sigmoïde
        ----------
        visible_v : vecteur des données d'entrée
        -------
        Return : vecteur des unités de sortie
        """
        return tf.nn.sigmoid(tf.matmul(visible_v, self.W) + self.b_bias)

    
    def _sortie_entree_(self, hidden_v):
        """Retourne la valeur des unités de sortie calculées à partir de la fonction sigmoïde
        ----------
        hidden_v : vecteur des données de sortie
        -------
        Return : vecteur des unités d'entrée '
        """
        return tf.nn.sigmoid(tf.matmul(hidden_v, tf.transpose(self.W)) + self.a_bias)

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

        grad_w_new = -(positive_grad - negative_grad) / tf.compat.v1.to_float(tf.shape(input)[0])
        grad_visible_bias_new = -(tf.reduce_mean(input - negative_visible_samp, 0))
        grad_hidden_bias_new = -(tf.reduce_mean(proba_h_sachant_v - negative_hidden_samp, 0))

        grads = [grad_w_new, grad_visible_bias_new, grad_hidden_bias_new]

        return grads
        
    
    def _training(self, loss=None, grads=None):
        """Sets up the training Ops.
        Applies the gradients to all trainable variables.
        If no loss is provided, gradients default to grads.
        Args:
            loss: loss to be minimized.
            grads: gradients to be minimized.
        Returns:
            train_op: the Op for training.
        """
        train_op = self.optimizer.apply_gradients(list(zip(grads, tf.compat.V1.trainable_variables())))
        return train_op


    def _reconstruct(self, visible_input):
        """ Reconstructs visible variables.
        Args:
            visible_input: visible_input to be reconstructed
        Returns:
            reconstruction
        """
        return self._sortie_entree_(sample_bernoulli(self._entree_sortie_(visible_input)))

    def _gibbs_sample_v_prime_given_v(self, visible_input, steps=500):
        """ Perform n-step Gibbs sampling chain in order to obtain the marginal distribution p(v|W,a,b) of the
        visible variables.
        Args:
            visible_input: visible_input to initialize gibbs chain
            steps: number of steps that Gibbs sampling chain is run for
        """
        v = visible_input
        for step in range(steps):
            v = sample_bernoulli(self._sortie_entree_(sample_bernoulli(self._entree_sortie_(v))))
        return v

    def _free_energy(self, v):
        """ FE(v) = −(aT)(v) − ∑_{i}log(1 + e^(b_{i} + W_{i}v))
        """
        return - tf.matmul(v, tf.expand_dims(self.a_bias, -1)) \
            - tf.reduce_sum(tf.math.log(1 + tf.exp(self.b_bias + tf.matmul(v, self.W))), axis=1)

    def _pseudo_log_likelihood(self):
        """ log(PL(v)) ≈ N * log(sigmoid(FE(v_{i}) − FE(v)))
        """
        v = sample_bernoulli(self.visible_vect)
        vi = tf.concat(
            [v[:, :self._idx_pll + 1], 1 - v[:, self._idx_pll + 1:self._idx_pll + 2], v[:, self._idx_pll + 2:]], 1)
        self._idx_pll = (self._idx_pll + 1) % self.visible_dim
        fe_x = self._free_energy(v)
        fe_xi = self._free_energy(vi)
        return tf.reduce_mean(tf.reduce_mean(
            self.visible_dim * tf.math.log(tf.nn.sigmoid(tf.clip_by_value(fe_xi - fe_x, -20, 20))), axis=0))

    def update_model(self, visible_input):
        """ Updates model parameters via single step of optimizer train_op.
        Args:
            visible_input: visible_input 
        """
        self.sess.run(self.train_op, feed_dict={self.visible_vect: visible_input})

    def eval_pll(self, visible_input):
        """ Evalulates pseudo_log_likelihood that model assigns to visible_input.
        Args:
            visible_input: visible_input 
        Returns:
            pseudo_log_likelihood
        """
        return self.sess.run(self.pll, feed_dict={self.visible_vect: visible_input})

    def eval_rec_error(self, visible_input):
        """ Evalulates single step reconstruction error of reconstruction of visible_input.
        Args:
            visible_input: visible_input
        Returns:
            reconstruction error
        """
        return self.sess.run(self.reconstruction_error, feed_dict={self.visible_vect: visible_input})

    def sample_v_marg(self, n_samples=100, size=784, epoch=0):
        """ This function samples images via Gibbs sampling chain in order to inspect the marginal distribution of the
        visible variables.
        Args:
            num_samples: an integer value representing the number of samples that will be generated by the model.
            size: size of visible samples.
            epoch: how many training epochs have occured before taking this sample.

        """
        cwd = os.getcwd()
        batch_v_noise = np.random.rand(n_samples, size)
        v_marg = self.sess.run(self.v_marg, feed_dict={self.visible_vect: batch_v_noise})
        v_marg = v_marg.reshape([n_samples, 28, 28])
        save_merged_images(images=v_marg, size=(10, 10), path=cwd)

    def save(self, file_path):
        """ Saves model.
        Args:
            file_path: path of file to save model in.
        """
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, file_path)

    def load(self, file_path):
        """ Loads model.
        Args:
            file_path: path of file to load model from.
        """
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, file_path)

    


