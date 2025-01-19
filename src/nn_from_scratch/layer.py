from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import signal


class Layer(ABC):
    def __init__(self, *kwargs):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input) -> Any: ...

    @abstractmethod
    def backward(self, output_gradient, lr) -> Any: ...


class Dense(Layer):
    def __init__(self, batch_size, input_size, output_size):
        # self.weights = np.random.randn(output_size, input_size)  # w[j][i]
        # self.bias = np.random.randn(output_size, 1)
        self.weights = self._he_init(input_size, output_size)
        self.bias = np.zeros((output_size, batch_size))

    def _he_init(self, input_size, output_size):
        stddev = np.sqrt(2.0 / input_size)
        return np.random.normal(0, stddev, (output_size, input_size))

    def _random_normal_init(self, input_size, output_size):
        return np.random.normal(0, 1, size=(output_size, input_size))

    def forward(self, input):
        # input (input_size, b)
        self.input = input
        # (output_size, b) = (output_size, input_size) @ (input_size, b)
        y = self.weights @ self.input + self.bias
        return y
        # return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, lr):
        # (output_size, input_size) = (output_size, b) @ (b, input_size)
        w_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= lr * w_gradient
        self.bias -= lr * output_gradient
        # (input_size, b) = (input_size, output_size) @ (output_size, b)
        input_gradient = self.weights.T @ output_gradient
        return input_gradient
        # return np.dot(self.weights.T, output_gradient)


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
        # self.kernels = np.random.rand(*self.kernel_shape)
        # self.kernels = np.random.normal(scale=1e-2, size=self.kernel_shape)
        self.kernels = self._he_init(*self.kernel_shape)
        self.biases = np.zeros(self.output_shape)
        # self.biases = np.random.rand(*self.output_shape)

    def _he_init(self, depth, input_depth, kernel_height, kernel_width):
        fan_in = input_depth * kernel_height * kernel_width  # Calculate fan-in
        stddev = np.sqrt(
            2.0 / fan_in
        )  # Standard deviation for He initialization
        return np.random.normal(
            0,
            stddev,
            (depth, input_depth, kernel_height, kernel_width),
        )  # Normal distribution

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


class Softmax(Layer):
    def forward(self, input):
        self.input = input
        nominator = np.exp(self.input)  # vector
        denominator = np.sum(nominator)  # scalar
        self.output = nominator / denominator  # vector
        assert self.output.shape == self.input.shape
        return nominator / denominator

    def backward(self, output_gradient, lr):
        # Assuming c classes
        # self.output.shape => (c, 1)
        n = self.output.size()  # number of reps for tiling next line
        # m => (c, c)
        m = np.tile(self.output, n)
        grad = np.dot(m * (np.identity(n) - m.T), output_gradient)
        """
            Remember the gradient propagates backwards
            It is a gradient w.r.t self.input (x) => dL/dX = dy/dx * dL/dy
            the latter term dL/dy is the output_gradient
            the grad variable will become the dL/dy (output_gradient) for
            the layer before.
            As softmax layer usually the last layer and the layer before it
            usually is a dense layer.
        """
        assert grad.shape == self.input.shape
        return grad


class BatchNorm1D(Layer):
    def __init__(self, d: int, momentum: float = 0.1, eps=1e-6):
        self.gamma = np.ones((d, 1))
        self.beta = np.zeros((d, 1))
        self.running_mu = np.zeros((d, 1))
        self.running_var = np.zeros((d, 1))
        self.momentum = momentum
        self.eps = eps

    def forward(self, input, train=True):
        self.input = input
        if train:
            mu = np.mean(input, axis=0, keepdims=True)
            sigma = np.sqrt(np.var(input, axis=0, keepdims=True) + self.eps)
            input_hat = (self.input - mu) / sigma
            self.running_mu = (
                1 - self.momentum
            ) * self.running_mu + self.momentum * mu
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * (sigma**2)
        else:
            mu = self.running_mu
            sigma = np.sqrt(self.running_var + self.eps)
            input_hat = (self.input - mu) / sigma
        self.input_hat = input_hat
        output = input_hat * self.gamma + self.beta
        assert output.shape == input.shape
        return output

    def backward(self, output_gradient, lr):
        # local gradient for gamma
        # gamma_gradient is essentially input_hat
        gamma_gradient = self.input_hat
        self.gamma -= lr * np.sum(output_gradient * gamma_gradient, axis=0)
        # local gradient for beta
        beta_gradient = 1 * output_gradient
        self.beta -= lr * np.sum(beta_gradient, axis=0)
        # return gradient w.r.t to input
        input_gradient = np.multiply(
            output_gradient, self.gamma / np.sqrt(self.running_var + self.eps)
        )
        assert input_gradient.shape == output_gradient.shape
        return input_gradient
