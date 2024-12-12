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


def ce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    print(y_pred)
    print(y_pred.shape)
    probs = np.exp(y_pred) / np.sum(np.exp(y_pred))
    print(probs)
    print(probs.shape)
    raise SystemExit
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss


def ce_loss_prime(y_true, y_pred):
    pass


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
        # y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        exp_y_pred = np.exp(y_pred)
        # probs = exp_y_pred / np.sum(exp_y_pred, axis=1).reshape(-1, 1)
        probs = exp_y_pred / np.sum(exp_y_pred)
        print("y_pred_prob", probs)
        loss = -np.sum(y_true * np.log(probs))
        # loss = -np.mean(np.sum(y_true * np.log(probs), axis=1))
        return loss

    @staticmethod
    def loss_prime(y_true, y_pred):
        exp_y_pred = np.exp(y_pred)
        probs = exp_y_pred / np.sum(exp_y_pred)
        loss_prime = probs - y_true
        return loss_prime
