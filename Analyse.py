from DBN import *
from DNN import *
from tools import *


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = lire_mnist()
    y_train = one_hot_encode(y_train, 10)
    y_test = one_hot_encode(y_test, 10)
    # Flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    dbn_analyse = DBN(3, X_train.shape[1], 128)
    dbn_analyse.train(X_train, 10, 1024, 0.5)
    dnn_analyse = DNN([X_train.shape[1], 128, 128, 128, 10])
    dnn_analyse.init_DNN_with_DBN(dbn_analyse)
    dnn_analyse.train(X_train, y_train, 100, 128, 0.2)