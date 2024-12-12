import numpy as np
import torch
import torch.nn as nn

from .activation import ReLU, Softmax
from .layer import Convolution, Dense, Reshape
from .loss import Loss


def create_batch(x, y, batch_size):
    n_batch = int(np.ceil(len(x) / batch_size))
    batch_x = np.array_split(x, n_batch)
    batch_y = np.array_split(y, n_batch)
    return batch_x, batch_y


def train(
    x_train,
    y_train,
    network,
    loss: Loss,
    epoches: int = 100,
    lr: np.float32 = np.float32(0.1),
):
    for e in range(epoches):
        tot_epoch_error = 0
        n_batch = 0
        n_images_per_batch = 0
        # shuffle data and then make batch for each epoch
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        shuffled_x = x_train[indices]
        shuffled_y = y_train[indices]
        shuffled_batch_x, shuffled_batch_y = create_batch(
            shuffled_x, shuffled_y, 32
        )

        # for i, (batch_x, batch_y) in enumerate(zip(x_train, y_train)):
        for _, (batch_x, batch_y) in enumerate(
            zip(shuffled_batch_x, shuffled_batch_y)
        ):
            tot_batch_error = 0
            n_images_per_batch = len(batch_x)
            for x, y in zip(batch_x, batch_y):
                output = x
                for layer in network:
                    output = layer.forward(output)

                y_true_probs = Softmax._softmax(y)
                my_loss = loss.loss(y_true_probs, output)

                # tot_batch_error += loss.loss(y, output)
                tot_batch_error += my_loss

                grad = loss.loss_prime(y_true_probs, output)

                for layer in network[::-1]:
                    grad = layer.backward(grad, lr)

            avg_batch_error = tot_batch_error / len(batch_x)
            print(f"{e+1}/{epoches} {avg_batch_error=}")
            tot_epoch_error += tot_batch_error
            n_batch += 1
        avg_epoch_error = tot_epoch_error / (n_batch * n_images_per_batch)
        print(f"{e+1}/{epoches} {avg_epoch_error=}")
        print("-" * 50)


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output
