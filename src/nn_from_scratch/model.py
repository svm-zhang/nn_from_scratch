import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Generator, Optional

import numpy as np

from .activation import ReLU
from .layer import (
    BatchNorm1D,
    Convolution,
    Dense,
    Layer,
    MaxPool,
    Parameter,
    Reshape,
)


class BaseModel(ABC):
    def __init__(self, *kwargs): ...

    @abstractmethod
    def parameters(self) -> Generator[Parameter, None, None]: ...

    @abstractmethod
    def add_layer(self, layer: Layer): ...

    @property
    def layers(self) -> list[Layer]: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    def __call__(self, input): ...

    @abstractmethod
    def train(self): ...

    @abstractmethod
    def eval(self): ...

    @abstractmethod
    def forward(self, input): ...

    @abstractmethod
    def backward(self, grad): ...

    @abstractmethod
    def state_dict(self) -> OrderedDict[str, Parameter]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict): ...


class CNNModel(BaseModel):
    # def __init__(self, input_shape):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_shape: int,
        ks: list[int],
        depths: list[int],
        paddings: list[int],
        fc_features: list[int],
        pool_method: str = "max",
        fs: Optional[list[int]] = None,
        strides: Optional[list[int]] = None,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.ks = ks
        self.depths = depths
        self.fs = fs
        self.strides = strides
        self.paddings = paddings
        self.fc_features = fc_features
        if pool_method not in ["max", "ave"]:
            raise ValueError(
                f"Invalid value for pooling method: {pool_method}"
            )
        self.pool_method = pool_method
        self.training = False
        self._layers = []
        self._named_parameters = {}
        self._build()

    def add_layer(self, layer: Layer):
        self._layers.append(layer)

    @property
    def layers(self) -> list[Layer]:
        return self._layers

    def parameters(self) -> Generator[Parameter, None, None]:
        for layer in self.layers:
            tmp = layer.parameters()
            if tmp is None:
                continue
            for param in tmp:
                yield param[1]

    def _build(self):
        assert len(self.ks) == len(self.depths)
        assert len(self.ks) == len(self.paddings)
        if self.fs is not None:
            assert len(self.ks) == len(self.fs)
        if self.strides is not None:
            assert len(self.ks) == len(self.strides)
        input_shape = self.input_shape
        for i in range(len(self.ks)):
            k = self.ks[i]
            depth = self.depths[i]
            padding = self.paddings[i]
            conv = Convolution(input_shape, k, depth, padding)
            self.add_layer(conv)
            input_shape = conv.output_shape
            # Add pooling layer if possible
            if self.fs is not None:
                f = self.fs[i]
                stride = self.strides[i] if self.strides is not None else None
                # filter size == -1 means skip
                if f != -1:
                    pool = MaxPool(
                        input_shape, f, method=self.pool_method, stride=stride
                    )
                    self.add_layer(pool)
                    input_shape = pool.output_shape
            relu = ReLU()
            self.add_layer(relu)
        fc_in = np.prod(input_shape)
        reshap = Reshape(input_shape, fc_in)
        self.add_layer(reshap)
        for i in range(len(self.fc_features)):
            fc_feature = self.fc_features[i]
            fc = Dense(fc_in, fc_feature)
            self.add_layer(fc)
            relu = ReLU()
            self.add_layer(relu)
            # batch_norm = BatchNorm1D(fc_feature)
            # self.add_layer(batch_norm)
            fc_in = fc_feature
        fc = Dense(fc_in, self.output_shape)
        self.add_layer(fc)

    def __call__(self, input):
        return self.forward(input)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, input):
        output = input
        for layer in self.layers:
            if isinstance(layer, BatchNorm1D):
                output = layer.forward(output, self.training)
            else:
                output = layer.forward(output)
        return output

    def backward(self, grad):
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        return grad

    def state_dict(self) -> OrderedDict[str, Parameter]:
        params = []
        for i, layer in enumerate(self.layers):
            tmp = layer.parameters()
            if tmp is None:
                continue
            params += [
                (f"{self.name}.{i}.{param[0]}", param[1]) for param in tmp
            ]
        return OrderedDict(params)

    def load_state_dict(self, state_dict):
        states = list(state_dict.items())
        for i, param in enumerate(self.parameters()):
            state = states[i]
            param.value = state[1].value


def save(obj, fspath) -> None:
    with open(fspath, "wb") as f:
        pickle.dump(obj, f)


def load(fspath):
    with open(fspath, "rb") as f:
        return pickle.load(f)
