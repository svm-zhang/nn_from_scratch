from typing import Any, Protocol

import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )
    return loss


def bce_loss_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / (y_pred * (1 - y_pred)) / np.size(y_true)


class Loss(Protocol):
    @staticmethod
    def loss(y_true_idx, logits) -> Any: ...

    @staticmethod
    def loss_prime(y_true_idx, logits) -> Any: ...


class MSELoss:
    @staticmethod
    def loss(y_true, y_pred) -> np.float32:
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def loss_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)


class BCELoss:
    @staticmethod
    def loss(y_true, y_pred) -> np.float32:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    @staticmethod
    def loss_prime(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / np.size(y_true)


class CELoss(Loss):
    @staticmethod
    def loss(y_true_idx, logits):
        # y_true: (b, )
        # logits: (b, c), c is the number of output classes or output shape
        # print(f"{logits.shape=} {y_true_idx.shape=}")
        # print(f"{y_true_idx=}")
        # true_class_logits: (b, 1)
        true_class_logits = logits[np.arange(len(logits)), y_true_idx]
        true_class_logits = true_class_logits.reshape(-1, 1)
        # print(true_class_logits)
        # log_denominator: (b, )
        log_denominator = np.log(np.sum(np.exp(logits), axis=-1))
        log_denominator = log_denominator.reshape(-1, 1)
        # print(f"{log_denominator.shape=}")
        # print(f"{true_class_logits.shape=}")
        # print(f"{y_pred=} {y_pred.shape=} {len(y_pred)=}")
        # true_class_logits = y_pred[y_true, 0]
        # print(f"{true_class_logits=}")
        loss = -true_class_logits + log_denominator
        # loss = -true_class_logits + np.log(np.sum(np.exp(logits)))
        return loss

    @staticmethod
    def loss_prime(y_true_idx, logits) -> np.ndarray:
        # logits: (b, c)
        # one_true_class: (b, c)
        one_true_class = np.zeros_like(logits)
        # print(f"{one_true_class.shape}")
        one_true_class[np.arange(len(logits)), y_true_idx] = 1
        softmax = np.exp(logits) / np.sum(
            np.exp(logits), axis=-1, keepdims=True
        )
        # print(f"{softmax.shape=}")
        loss_grad = (softmax - one_true_class) / logits.shape[0]
        # print(f"{loss_grad.shape=}")
        assert loss_grad.shape == logits.shape
        return loss_grad
        # ones_true_class = np.zeros_like(y_pred)
        ones_true_class[y_true, 0] = 1
        # print("ones_true_class", ones_true_class)
        softmax = np.exp(y_pred) / np.exp(y_pred).sum(0, keepdims=True)
        # print("softmax", softmax, softmax.shape)
        # print(-ones_true_class + softmax)
        loss_prime = (-ones_true_class + softmax) / y_pred.shape[0]
        # print(f"{loss_prime.shape=}")
        # print("Ending loss_prime")
        return loss_prime
