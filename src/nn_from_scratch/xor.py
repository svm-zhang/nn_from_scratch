import numpy as np

from .activation import Sigmoid
from .layer import Dense
from .loss import BCELoss
from .model import predict, train


def solve_xor():
    x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    # nn = [Dense(2, 3), Tanh(), Dense(3, 1), Tanh()]
    nn = [Dense(2, 3), Sigmoid(), Dense(3, 1), Sigmoid()]

    epoches = 20000
    lr = 0.04

    loss = BCELoss()
    train(x_train, y_train, nn, loss, epoches, lr)

    x_test = np.reshape([[1, 0]], (2, 1))
    y_test = np.reshape([[0]], (1, 1))
    y_pred = predict(nn, x_test)
    print(y_pred)
