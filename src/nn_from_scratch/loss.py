from typing import Protocol

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
    def loss(y_true, y_pred) -> np.float32: ...

    @staticmethod
    def loss_prime(y_true, y_pred): ...


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


class CELoss:
    @staticmethod
    def loss(y_true, y_pred) -> np.float32:
        # print(f"{y_true=}")
        # print(f"{y_pred=} {y_pred.shape=} {len(y_pred)=}")
        true_class_logits = y_pred[y_true, 0]
        # print(f"{true_class_logits=}")
        loss = -true_class_logits + np.log(np.sum(np.exp(y_pred)))
        return loss

    @staticmethod
    def loss_prime(y_true, y_pred):
        ones_true_class = np.zeros_like(y_pred)
        # print("I am get dl/dz")
        # print(y_true)
        # print(f"{ones_true_class=}")
        ones_true_class[y_true, 0] = 1
        # print("ones_true_class", ones_true_class)
        softmax = np.exp(y_pred) / np.exp(y_pred).sum(0, keepdims=True)
        # print("softmax", softmax, softmax.shape)
        # print(-ones_true_class + softmax)
        loss_prime = (-ones_true_class + softmax) / y_pred.shape[0]
        # print(f"{loss_prime.shape=}")
        # print("Ending loss_prime")
        return loss_prime

    @staticmethod
    def loss_old(y_true, y_pred) -> np.float32:
        # print("y_true", y_true)
        # print("y_pred", y_pred)
        # y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        exp_y_pred = np.exp(y_pred - np.max(y_pred))
        # probs = exp_y_pred / np.sum(exp_y_pred, axis=1).reshape(-1, 1)
        probs = exp_y_pred / np.sum(exp_y_pred)
        if (probs == 0).any():
            raise SystemExit
        # print("y_pred_prob", probs)
        # print("log y_pred_prob", np.log(probs))
        # print("product y_true log_y_pred_prob", y_true * np.log(probs))
        loss = -np.sum(np.multiply(y_true, np.log(probs)))
        return loss

    @staticmethod
    def loss_prime_old(y_true, y_pred):
        exp_y_pred = np.exp(y_pred - np.max(y_pred))
        probs = exp_y_pred / np.sum(exp_y_pred)
        loss_prime = probs - y_true
        return loss_prime
