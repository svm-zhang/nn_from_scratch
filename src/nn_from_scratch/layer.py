from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import signal


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input) -> Any: ...

    @abstractmethod
    def backward(self, output_gradient, lr) -> Any: ...


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)  # w[j][i]
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, lr):
        w_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= lr * w_gradient
        self.bias -= lr * output_gradient
        return np.dot(self.weights.T, output_gradient)


class Convolution(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth  # number of kernels / outputs
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (
            depth,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        )
        self.kernel_shape = (
            self.depth,
            self.input_depth,
            kernel_size,
            kernel_size,
        )
        self.kernels = np.random.rand(*self.kernel_shape)
        self.biases = np.random.rand(*self.output_shape)

    def forward(self, input):
        # Y = B + X * K
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i, j], mode="valid"
                )
        return self.output

    def backward(self, output_gradient, lr):
        kernel_grad = np.zeros(self.kernel_shape)
        input_grad = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernel_grad[i, j] = signal.correlate2d(
                    self.input[j], output_gradient[i], mode="valid"
                )
                input_grad[j] += signal.convolve2d(
                    output_gradient[i], self.kernels[i, j], mode="full"
                )

        self.kernels -= lr * kernel_grad
        self.biases -= lr * output_gradient

        return input_grad


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, lr):
        return np.reshape(output_gradient, self.input_shape)

    pass
