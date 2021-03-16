import numpy as np

def sample_bernoulli(proba):
    """ Prend un échantillon d'une distribution de Bernoulli
    Arguments:
        proba: Distribution de proba de laquelle prendre un échantillon
    Return:
        sample: échantillon de la distribution
    """
    sample = (np.random.random(proba.shape) < proba)*1
    return sample