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
    def __init__(
        self, params, lr, betas: tuple[float, float] = (0.9, 0.999), eps=1e-6
    ):
        super().__init__(params, lr)
        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self.t = 0
        self.vs = [np.zeros_like(param.grad) for param in self.params]
        self.ms = [np.zeros_like(param.grad) for param in self.params]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            assert isinstance(param, Parameter)
            if param.grad is None:
                continue
            self.ms[i] = (
                self.beta_1 * self.ms[i] + (1 - self.beta_1) * param.grad
            )
            mt_hat = self.ms[i] / (1 - self.beta_1**self.t)
            self.vs[i] = self.beta_2 * self.vs[i] + (
                1 - self.beta_2
            ) * np.power(param.grad, 2)
            vt_hat = self.vs[i] / (1 - self.beta_2**self.t)
            param.value -= (self.lr * mt_hat) / (np.sqrt(vt_hat) + self.eps)
