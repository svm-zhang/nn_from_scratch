from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np

from .model import Parameter


class optimizer(ABC):
    def __init__(self, params, lr, *kwargs):
        self.params = list(params)
        self.lr = lr
        self.state = defaultdict(list)

    @abstractmethod
    def step(self): ...

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    @abstractmethod
    def state_dict(self) -> Any: ...

    @abstractmethod
    def load_state_dict(self, state_dict): ...


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

    def state_dict(self):
        return {
            "state": [{i: self.vs[i]} for i in range(len(self.vs))],
            "lr": self.lr,
            "momentum": self.momentum,
        }

    def load_state_dict(self, state_dict):
        states = state_dict.get("state")
        for i in range(len(states)):
            state = states[i]
            self.vs[i] = state[i]
        self.lr = state_dict.get("lr")
        self.momentum = state_dict.get("momentum")


class Adam(optimizer):
    def __init__(
        self, params, lr, betas: tuple[float, float] = (0.9, 0.999), eps=1e-8
    ):
        super().__init__(params, lr)
        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self.ts = [0 for _ in range(len(self.params))]
        self.vs = [np.zeros_like(param.grad) for param in self.params]
        self.ms = [np.zeros_like(param.grad) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            assert isinstance(param, Parameter)
            if param.grad is None:
                continue
            self.ts[i] += 1
            self.ms[i] = (
                self.beta_1 * self.ms[i] + (1 - self.beta_1) * param.grad
            )
            mt_hat = self.ms[i] / (1 - self.beta_1 ** self.ts[i])
            self.vs[i] = self.beta_2 * self.vs[i] + (
                1 - self.beta_2
            ) * np.power(param.grad, 2)
            vt_hat = self.vs[i] / (1 - self.beta_2 ** self.ts[i])
            param.value -= (self.lr * mt_hat) / (np.sqrt(vt_hat) + self.eps)

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": [
                {
                    "step": self.ts[i],
                    "exp_avg": self.vs[i],
                    "exp_avg_sq": self.ms[i],
                }
                for i in range(len(self.params))
            ],
            "lr": self.lr,
            "betas": (self.beta_1, self.beta_2),
            "eps": self.eps,
        }

    def load_state_dict(self, state_dict):
        states = state_dict.get("state")
        for i in range(len(states)):
            state = states[i]
            self.ts[i] = state.get("step")
            self.vs[i] = state.get("exp_avg")
            self.ms[i] = state.get("exp_avg_sq")
        self.beta_1, self.beta_2 = state_dict.get("betas")
        self.lr = state_dict.get("lr")
        self.eps = state_dict.get("eps")
