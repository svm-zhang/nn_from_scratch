from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import signal

# TODO:
# 1. Add padding. [DONE]
# 2. Max pooling
# 3. BatchNormalization2D
# 4. Adam SGD
# 5. Use np.float32 [DONE]


class Layer(ABC):
    def __init__(self, *kwargs):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input) -> Any: ...

    @abstractmethod
    def backward(self, output_gradient, lr) -> Any: ...


class Dense(Layer):
    def __init__(self, input_size, output_size):
        # weights: (output_size, input_size)
        self.weights = self._he_init(input_size, output_size).astype(
            np.float32
        )
        # bias: (b, output_size)
        self.bias = np.zeros((1, output_size)).astype(np.float32)

    def _he_init(self, input_size, output_size):
        stddev = np.sqrt(2.0 / input_size)
        return np.random.normal(0, stddev, (output_size, input_size))

    def _random_normal_init(self, input_size, output_size):
        return np.random.normal(0, 1, size=(output_size, input_size))

    def _random_init(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)  # w[j][i]
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        # input (b, input_size)
        self.input = input
        # (b, output_size) = (b, input_size) @ (output_size, input_size).T
        y = self.input @ self.weights.T + self.bias
        return y

    def backward(self, output_gradient, lr):
        # output_gradient: (b, output_size)
        # (output_size, input_size) = (output_size, b) @ (b, input_size)
        # w_gradient = np.dot(output_gradient, self.input)
        w_gradient = output_gradient.T @ self.input
        b_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        self.weights -= lr * w_gradient
        self.bias -= lr * b_gradient
        # (b, input_size) = (b, output_size) @ (output_size, input_size)
        input_gradient = output_gradient @ self.weights
        return input_gradient


class Convolution(Layer):
    def __init__(self, input_shape, kernel_size, depth, padding: int = 0):
        input_depth, input_height, input_width = input_shape
        self.padding_x = padding
        self.padding_y = padding
        self.depth = depth  # number of kernels / outputs
        self.input_depth = input_depth
        # output height and width accounting for padding
        o_h = input_height + self.padding_x * 2 - kernel_size + 1
        o_w = input_width + self.padding_y * 2 - kernel_size + 1
        self.output_shape = (depth, o_h, o_w)
        self.kernel_size = kernel_size
        self.kernel_shape = (
            self.depth,
            self.input_depth,
            kernel_size,
            kernel_size,
        )
        self.kernels = self._he_init(*self.kernel_shape).astype(np.float32)
        self.biases = np.zeros(self.output_shape).astype(np.float32)

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
        self.batch_size = input.shape[0]
        self.input = input
        input = np.pad(
            input,
            pad_width=(
                (0, 0),
                (0, 0),
                (self.padding_x, self.padding_x),
                (self.padding_y, self.padding_y),
            ),
            mode="constant",
            constant_values=0,
        )
        output_shape = (self.batch_size, *self.output_shape)
        self.output = np.zeros(output_shape)
        for b in range(self.batch_size):  # first dimension is the batch_size
            for i in range(self.depth):
                self.output[b, i] = self.biases[i]
                for j in range(self.input_depth):
                    cross_correlation = signal.correlate2d(
                        input[b, j], self.kernels[i, j], mode="valid"
                    )
                    self.output[b, i] += cross_correlation
        return self.output

    def backward(self, output_gradient, lr):
        # (cout, cin, k, k)
        kernel_grad = np.zeros(self.kernel_shape)
        bias_grad = np.zeros(self.output_shape)
        # (b, cin, h, w)
        input_grad = np.zeros(self.input.shape)

        for b in range(self.batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    kernel_grad[i, j] += signal.correlate2d(
                        self.input[b, j], output_gradient[b, i], mode="valid"
                    )
                    bias_grad[i] += output_gradient[b, i]
                    full_convolve = signal.convolve2d(
                        output_gradient[b, i], self.kernels[i, j], mode="full"
                    )
                    if self.padding_x != 0:
                        full_convolve = full_convolve[
                            self.padding_x : -self.padding_x,
                            self.padding_y : -self.padding_y,
                        ]
                    input_grad[b, j] += full_convolve

        self.kernels -= lr * kernel_grad
        self.biases -= lr * bias_grad
        return input_grad


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, (input.shape[0], self.output_shape))

    def backward(self, output_gradient, lr):
        _ = lr
        batch_size = output_gradient.shape[0]
        cout, h, w = self.input_shape
        output_shape = (batch_size, cout, h, w)
        return np.reshape(output_gradient, output_shape)


class Softmax(Layer):
    def forward(self, input):
        self.input = input
        nominator = np.exp(self.input)  # vector
        denominator = np.sum(nominator, axis=-1, keepdims=True)  # scalar
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
        self.gamma = np.ones((1, d)).astype(np.float32)
        self.beta = np.zeros((1, d)).astype(np.float32)
        self.running_mu = np.zeros((1, d)).astype(np.float32)
        self.running_var = np.zeros((1, d)).astype(np.float32)
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
