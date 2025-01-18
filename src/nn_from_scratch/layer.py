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
    def __init__(self, input_size, output_size):
        # self.weights = np.random.randn(output_size, input_size)  # w[j][i]
        # self.bias = np.random.randn(output_size, 1)
        # self.weights = np.random.normal(
        #     scale=1e-2, size=(output_size, input_size)
        # )
        self.weights = self._he_init(input_size, output_size)
        self.bias = np.zeros((output_size, 1))

    def _he_init(self, input_size, output_size):
        stddev = np.sqrt(2.0 / input_size)
        return np.random.normal(0, stddev, (output_size, input_size))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, lr):
        # print("I am in Dense.backward")
        # print(f"{output_gradient.shape=}")
        # print(f"{self.input.shape=}")
        # print(f"{self.weights.shape=}")
        w_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= lr * w_gradient
        self.bias -= lr * output_gradient
        # (output_size, input_size) @ (output_size, 1) => (input_size, 1)
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


class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, lr):
        relu_grad = (self.input > 0).astype(self.input.dtype)
        # print("I am in ReLU backward")
        # print(f"{output_gradient.shape=}")
        # print(f"{self.input.shape=}")
        # element-wise multiply
        return np.multiply(output_gradient, relu_grad)
