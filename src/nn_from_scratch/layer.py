from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from scipy import signal

from .initialization import he_normal_init

# TODO:
# 3. BatchNormalization2D


class Parameter:
    def __init__(self, value, requires_grad):
        self._value = value
        self._grad = None
        if requires_grad:
            self._grad = np.zeros_like(self._value).astype(value.dtype)

    def zero_grad(self):
        if self._grad is not None:
            self._grad = np.zeros_like(self._grad).astype(self.value.dtype)

    @property
    def value(self) -> np.ndarray:
        return self._value

    @property
    def grad(self) -> np.ndarray | None:
        return self._grad

    @grad.setter
    def grad(self, v):
        if self._grad is None:
            return
        assert v.shape == self._grad.shape
        self._grad = v

    @value.setter
    def value(self, v) -> None:
        assert v.shape == self._value.shape
        self._value = v


class Layer(ABC):
    def __init__(self, *kwargs):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input) -> Any: ...

    @abstractmethod
    def backward(self, output_gradient) -> Any: ...

    @abstractmethod
    def parameters(self) -> list[tuple[str, Parameter]] | None: ...

    @property
    def name(self):
        return self.__class__.__name__.lower()


class Dense(Layer):
    def __init__(self, input_size, output_size):
        # weights: (output_size, input_size)
        self._weights: Parameter = Parameter(
            he_normal_init((input_size,), output_size).astype(np.float32),
            requires_grad=True,
        )
        # bias: (b, output_size)
        self._bias: Parameter = Parameter(
            np.zeros((1, output_size)).astype(np.float32),
            requires_grad=True,
        )

    def _he_init(self, input_size, output_size):
        stddev = np.sqrt(2.0 / input_size)
        return np.random.normal(0, stddev, (output_size, input_size))

    @property
    def weights(self) -> Parameter:
        return self._weights

    @property
    def bias(self) -> Parameter:
        return self._bias

    def parameters(self) -> list[tuple[str, Parameter]]:
        return [
            (f"{self.name}.weights", self.weights),
            (f"{self.name}.bias", self.bias),
        ]

    def forward(self, input):
        # input (b, input_size)
        self.input = input
        # (b, output_size) = (b, input_size) @ (output_size, input_size).T
        y = self.input @ self.weights.value.T + self.bias.value
        return y

    def backward(self, output_gradient):
        # output_gradient: (b, output_size)
        # (output_size, input_size) = (output_size, b) @ (b, input_size)
        input_gradient = output_gradient @ self.weights.value
        self.weights.grad = output_gradient.T @ self.input
        self.bias.grad = np.sum(output_gradient, axis=0, keepdims=True)

        # (b, input_size) = (b, output_size) @ (output_size, input_size)
        return input_gradient


class Convolution(Layer):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        kernel_size: int,
        depth: int,
        padding: int = 0,
    ):
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
        self.weights = Parameter(
            he_normal_init(
                (input_depth, kernel_size, kernel_size), depth
            ).astype(np.float32),
            requires_grad=True,
        )
        self.bias = Parameter(
            np.zeros(self.output_shape).astype(np.float32),
            requires_grad=True,
        )

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

    def parameters(self) -> list[tuple[str, Parameter]]:
        return [
            (f"{self.name}.weights", self.weights),
            (f"{self.name}.bias", self.bias),
        ]

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
                self.output[b, i] = self.bias.value[i]
                for j in range(self.input_depth):
                    cross_correlation = signal.correlate2d(
                        input[b, j], self.weights.value[i, j], mode="valid"
                    )
                    self.output[b, i] += cross_correlation
        return self.output

    def backward(self, output_gradient):
        # (cout, cin, k, k)
        # kernel_grad = np.zeros(self.kernel_shape)
        # bias_grad = np.zeros(self.output_shape)
        # (b, cin, h, w)
        input_grad = np.zeros(self.input.shape)

        assert self.weights.grad is not None
        assert self.bias.grad is not None
        for b in range(self.batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    self.weights.grad[i, j] += signal.correlate2d(
                        self.input[b, j], output_gradient[b, i], mode="valid"
                    )
                    self.bias.grad[i] += output_gradient[b, i]
                    full_convolve = signal.convolve2d(
                        output_gradient[b, i],
                        self.weights.value[i, j],
                        mode="full",
                    )
                    if self.padding_x != 0:
                        full_convolve = full_convolve[
                            self.padding_x : -self.padding_x,
                            self.padding_y : -self.padding_y,
                        ]
                    input_grad[b, j] += full_convolve

        return input_grad


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def parameters(self) -> None:
        pass

    def forward(self, input):
        return np.reshape(input, (input.shape[0], self.output_shape))

    def backward(self, output_gradient):
        batch_size = output_gradient.shape[0]
        cout, h, w = self.input_shape
        output_shape = (batch_size, cout, h, w)
        return np.reshape(output_gradient, output_shape)


class Softmax(Layer):
    def parameters(self) -> None:
        pass

    def forward(self, input):
        self.input = input
        nominator = np.exp(self.input)  # vector
        denominator = np.sum(nominator, axis=-1, keepdims=True)  # scalar
        self.output = nominator / denominator  # vector
        assert self.output.shape == self.input.shape
        return nominator / denominator

    def backward(self, output_gradient):
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
        self.gamma = Parameter(
            np.ones((1, d)).astype(np.float32), requires_grad=True
        )
        self.beta = Parameter(
            np.zeros((1, d)).astype(np.float32), requires_grad=True
        )
        self.running_mu = Parameter(
            np.zeros((1, d)).astype(np.float32),
            requires_grad=False,
        )
        self.running_var = Parameter(
            np.zeros((1, d)).astype(np.float32),
            requires_grad=False,
        )
        self.momentum = momentum
        self.eps = eps

    def parameters(self) -> list[tuple[str, Parameter]]:
        return [
            (f"{self.name}.weights", self.gamma),
            (f"{self.name}.bias", self.beta),
            (f"{self.name}.running_mu", self.running_mu),
            (f"{self.name}.running_var", self.running_var),
        ]

    def forward(self, input, train=True):
        self.input = input
        if train:
            mu = np.mean(input, axis=0, keepdims=True)
            sigma = np.sqrt(np.var(input, axis=0, keepdims=True) + self.eps)
            input_hat = (self.input - mu) / sigma
            self.running_mu.value = (
                1 - self.momentum
            ) * self.running_mu.value + self.momentum * mu
            self.running_var.value = (
                1 - self.momentum
            ) * self.running_var.value + self.momentum * (sigma**2)
        else:
            mu = self.running_mu.value
            sigma = np.sqrt(self.running_var.value + self.eps)
            input_hat = (self.input - mu) / sigma
        self.input_hat = input_hat
        output = input_hat * self.gamma.value + self.beta.value
        assert output.shape == input.shape
        return output

    def backward(self, output_gradient):
        input_gradient = np.multiply(
            output_gradient,
            self.gamma.value / np.sqrt(self.running_var.value + self.eps),
        )
        assert input_gradient.shape == output_gradient.shape
        # local gradient for gamma
        # gamma_gradient is essentially input_hat
        self.gamma.grad = np.sum(
            output_gradient * self.input_hat, axis=0, keepdims=True
        )
        # local gradient for beta
        self.beta.grad = np.sum(output_gradient, axis=0, keepdims=True)
        # return gradient w.r.t to input
        return input_gradient


class MaxPool(Layer):
    def __init__(
        self,
        input_shape,
        f: int,
        method: str,
        stride: Optional[int] = None,
    ):
        self.method = method
        self.f = f
        if stride is None:
            self.stride = f
        else:
            self.stride = stride
        self.input_shape = input_shape
        c, h, w = input_shape
        o_h = (w - self.f) // self.stride + 1
        o_w = (h - self.f) // self.stride + 1
        self.output_shape = (c, o_h, o_w)

    def parameters(self) -> None:
        return

    def forward(self, input):
        self.input = input
        assert input.ndim == len(input.strides)
        stride_b, stride_c, stride_h, stride_w = input.strides
        view_shape = (
            input.shape[:2] + self.output_shape[-2:] + (self.f, self.f)
        )
        strides = (
            stride_b,
            stride_c,
            stride_h * self.stride,
            stride_w * self.stride,
            stride_h,
            stride_w,
        )
        input_view = np.lib.stride_tricks.as_strided(
            input, view_shape, strides=strides, writeable=False
        )
        output = np.nanmax(input_view, axis=(4, 5), keepdims=True)
        self.pos = np.where(output == input_view, 1, 0)
        output = np.squeeze(output, axis=(4, 5))
        self.output_shape = output.shape

        return output

    def backward(self, output_gradient):
        input_gradient = np.zeros(self.input.shape)
        ib, ic, ih, iw, iy, ix = np.where(self.pos == 1)
        ih2 = ih * self.stride
        iw2 = iw * self.stride
        ih2 += iy
        iw2 += ix
        values = output_gradient[ib, ic, ih, iw].flatten()
        input_gradient[ib, ic, ih2, iw2] = values
        return input_gradient
