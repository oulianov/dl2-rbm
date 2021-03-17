import numpy as np

from RBM import RBM
from DBN import DBN
from tools import *

def dsigmoid(x):
    z = sigmoid(x)
    return z*(1-z)

def softmax(x):
    z = np.exp(x)
    return z/np.sum(z,axis=1, keepdims=True)

def dsoftmax(x):
    z = softmax(x)
    return -z.T@z + np.diagflat(z)

def dloss_x(x,y):
    return softmax(x) - np.diagflat(y)

def dloss_w(x_ant, x_post,y):
    pass

    
class DNN(DBN):
    '''Creates a Deep Neural using RBM'''

    def __init__(self, nb_couches, visible_dim, hidden_dim,K):
        pass

    def calcul_softmax():
        pass 

    def fine_tuning(self,X,y):
        pass





if __name__=='__main__':
    vetor = np.array([[0,1,-1]])
    print(dsoftmax(vetor))


