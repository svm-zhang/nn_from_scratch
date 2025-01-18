from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2

from .activation import Sigmoid
from .layer import Convolution, Dense, ReLU, Reshape
from .loss import BCELoss, CELoss
from .model import predict, train


def mnist():
    transform = v2.Compose(
        [
            v2.PILToTensor(),
        ]
    )
    mnist_data = torchvision.datasets.MNIST(
        "./mnist/", download=True, train=True, transform=transform
    )
    return mnist_data


def extract_tensor(dataloader):
    xs = []
    ys = []
    for i, (x, y) in enumerate(dataloader):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).detach().numpy(), torch.cat(ys).detach().numpy()


def preprocess(x, y, size):
    zero_index = np.where(y == 0)[0][:size]
    one_index = np.where(y == 1)[0][:size]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np.eye(2)[y]
    y = y.reshape(len(y), 2, 1)
    return x, y


def subset(dataset, indices, n_per_label):
    counter = defaultdict(int)
    num_classes = len(torch.unique(dataset.targets))
    sub_xs = []
    sub_ys = []
    for idx in indices:
        x, y = (
            dataset.data[idx],
            dataset.targets[idx],
        )
        label = y.item()
        if counter[label] < n_per_label:
            x = x.float() / 255.0
            x = torch.reshape(x, (1, x.shape[0], x.shape[1])).detach().numpy()
            y = F.one_hot(y, num_classes)
            y = y.view(-1, 1).detach().numpy()
            sub_xs.append(x)
            sub_ys.append(y)
            counter[label] += 1

    sub_x = np.array(sub_xs)
    sub_y = np.array(sub_ys)
    return sub_x, sub_y


def create_batch(x, y, batch_size):
    n_batch = int(np.ceil(len(x) / batch_size))
    batch_x = np.array_split(x, n_batch)
    batch_y = np.array_split(y, n_batch)
    return batch_x, batch_y


def solve_mnist():
    mnist_data = mnist()
    seed = torch.Generator().manual_seed(2024)

    train_dataset, test_dataset = torch.utils.data.random_split(
        mnist_data, [0.7, 0.3], generator=seed
    )
    sub_x_train, sub_y_train = subset(mnist_data, train_dataset.indices, 256)
    sub_x_test, sub_y_test = subset(mnist_data, test_dataset.indices, 32)
    # batch_x_train, batch_y_train = create_batch(sub_x_train, sub_y_train, 32)

    n_kernel = 3
    kernel_size = 3

    network = [
        Convolution((1, 28, 28), kernel_size, n_kernel),
        # Sigmoid(),
        ReLU(),
        # Convolution((3, 24, 24), kernel_size, 1),
        Reshape((n_kernel, 26, 26), (n_kernel * 26 * 26, 1)),
        Dense(n_kernel * 26 * 26, 128),
        ReLU(),
        # Sigmoid(),
        Dense(128, 10),
        ReLU(),
        # Sigmoid(),
    ]
    epoch = 120
    lr = 0.001

    # loss = BCELoss()
    loss = CELoss()

    train(sub_x_train, sub_y_train, network, loss, epoch, lr)
    # train(batch_x_train, batch_y_train, cnn, loss, epoch, lr)

    correct = 0
    i = 0
    for x, y in zip(sub_x_test, sub_y_test):
        output = predict(network, x)
        pred = np.argmax(output)
        truth = np.argmax(y)
        if pred == truth:
            correct += 1
        else:
            print(f"{pred=} {truth=}")
        i += 1

    print(correct)
    print(len(sub_y_test))
    print(correct / len(sub_y_test))
