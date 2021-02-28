import scipy.io
import os
import string
import numpy as np

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

class RBM(object):
    def __init__(self, visible_dim, hidden_dim):
        
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        self.weights = 0.01 * np.random.randn(self.visible_dim, self.hidden_dim)
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)


    def entree_sortie_(self, data):
    """Retourne la valeur des unités de sortie calculées à partir de la fonction sigmoïde
    ----------
    data : données d'entrée
    -------
    Return : matrice des unités de sortie
    """
        return 0


