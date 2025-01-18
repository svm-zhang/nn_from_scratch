import numpy as np

from .layer import Layer


class Activation(Layer):
    def __init__(self, f, f_prime):
        self.f = f
        self.f_prime = f_prime

    def forward(self, input):
        self.input = input
        return self.f(self.input)

    def backward(self, output_gradient, lr: np.float32):
        # For activation layer, there are no parameters to train..
        _ = lr
        # element-wise multiplication
        return np.multiply(output_gradient, self.f_prime(self.input))


class Tanh(Activation):
    def __init__(self):
        super().__init__(Tanh._tanh, Tanh._tanh_prime)

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _tanh_prime(x):
        return 1 - np.tanh(x) ** 2


class ReLU(Activation):
    def __init__(self):
        super().__init__(ReLU._relu, ReLU._relu_prime)

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_prime(x):
        return (x > 0).astype(x.dtype)
        # return np.where(x > 0, 1, 0).astype(np.float32)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(Sigmoid._sigmoid, Sigmoid._sigmoid_prime)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_prime(x):
        exp_neg_x = np.exp(-x)
        return exp_neg_x / ((1 + exp_neg_x) ** 2)
