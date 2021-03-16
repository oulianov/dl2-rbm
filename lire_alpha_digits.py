import numpy as np 
import scipy 
import string 
import os


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
    mat = scipy.io.loadmat(cwd + "/data/binaryalphadigs.mat")

    if type(caractere) == int and caractere < 36 and caractere >= 0:
        data = mat.get("dat")[caractere]
    elif type(caractere) == str and len(caractere) == 1:
        data = mat.get("dat")[10 + string.ascii_uppercase.index(caractere)]
    else:
        raise ValueError("Data not available for that caracter")
    data = np.array([i for i in data])
    im_shape = data.shape[1:]
    data = data.reshape(data.shape[0], -1)
    return data, im_shape