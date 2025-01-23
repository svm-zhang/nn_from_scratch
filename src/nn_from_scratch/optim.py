from abc import ABC, abstractmethod

import numpy as np

from .model import Parameter


class optimizer(ABC):
    def __init__(self, params, lr, *kwargs):
        self.params = params
        self.lr = lr

    @abstractmethod
    def step(self): ...

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


class SGD(optimizer):
    def __init__(self, params, lr, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.vs = [np.zeros_like(param.grad) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            assert isinstance(param, Parameter)
            if param.grad is None:
                continue
            if self.momentum == 0:
                self.vs[i] = param.grad
            else:
                self.vs[i] = self.momentum * self.vs[i] + param.grad
            param.value -= self.lr * self.vs[i]


class Adam(optimizer):
    def __init__(self, params, lr, betas):
        super().__init__(params, lr)
        self.beta_1, self.beta_2 = betas

    def step(self):
        pass
