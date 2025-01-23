from abc import ABC, abstractmethod

import numpy as np

from .activation import ReLU
from .layer import BatchNorm1D, Convolution, Dense, Layer, Parameter, Reshape


class BaseModel(ABC):
    def __init__(self, *kwargs): ...

    @abstractmethod
    def parameters(self) -> list[Parameter]: ...

    @abstractmethod
    def add_layer(self, layer: Layer): ...

    @property
    def layers(self) -> list[Layer]: ...

    @abstractmethod
    def forward(self, input, train): ...

    @abstractmethod
    def backward(self, grad): ...

    @abstractmethod
    def train(self, input): ...


class CNNModel(BaseModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self._layers = []
        self._named_parameters = {}

    def add_layer(self, layer: Layer):
        self._layers.append(layer)

    @property
    def layers(self) -> list[Layer]:
        return self._layers

    def parameters(self) -> list[Parameter]:
        params = []
        for layer in self.layers:
            tmp = layer.parameters()
            if tmp is None:
                continue
            params += [param for param in tmp]
        return params

    def forward(self, input, train: bool):
        output = input
        for layer in self.layers:
            if isinstance(layer, BatchNorm1D):
                output = layer.forward(output, train)
            else:
                output = layer.forward(output)
        return output

    def train(self, input):
        return self.forward(input, train=True)

    def predict(self, input):
        return self.forward(input, train=False)

    def backward(self, grad):
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        return grad


def build_cnn_model(
    input_shape: tuple[int, int, int],
    output_shape: int,
    ks: list[int],
    depths: list[int],
    paddings: list[int],
    fc_features: list[int],
):
    assert len(ks) == len(depths)
    assert len(ks) == len(paddings)
    cnn = CNNModel(input_shape)
    print(f"{input_shape=}")
    for i in range(len(ks)):
        k = ks[i]
        depth = depths[i]
        padding = paddings[i]
        conv = Convolution(input_shape, k, depth, padding)
        cnn.add_layer(conv)
        input_shape = conv.output_shape
        relu = ReLU()
        cnn.add_layer(relu)
        print(f"{input_shape=}")
    fc_in = np.prod(input_shape)
    reshap = Reshape(input_shape, fc_in)
    cnn.add_layer(reshap)
    print("Add Dense")
    for i in range(len(fc_features)):
        fc_feature = fc_features[i]
        fc = Dense(fc_in, fc_feature)
        cnn.add_layer(fc)
        relu = ReLU()
        cnn.add_layer(relu)
        batch_norm = BatchNorm1D(fc_feature)
        cnn.add_layer(batch_norm)
        fc_in = fc_feature
        print(f"{fc_in=}")
    fc = Dense(fc_in, output_shape)
    cnn.add_layer(fc)

    return cnn
