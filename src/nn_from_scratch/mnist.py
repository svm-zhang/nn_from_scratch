import numpy as np
import torch
import torchvision
from torchvision.transforms import v2

from .activation import Sigmoid
from .layer import Convolution, Dense, Reshape
from .loss import BCELoss
from .model import predict, train


def mnist():
    transform = v2.Compose([v2.PILToTensor()])
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


def solve_mnist():
    mnist_data = mnist()
    seed = torch.Generator().manual_seed(2024)
    train_dataset, test_dataset = torch.utils.data.random_split(
        mnist_data, [0.7, 0.3], generator=seed
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)
    x_train, y_train = extract_tensor(train_loader)
    x_test, y_test = extract_tensor(test_loader)

    x_train, y_train = preprocess(x_train, y_train, 300)
    x_test, y_test = preprocess(x_test, y_test, 100)

    cnn = [
        Convolution((1, 28, 28), 3, 5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid(),
    ]
    epoch = 20
    lr = 0.01

    loss = BCELoss()

    train(x_train, y_train, cnn, loss, epoch, lr)

    correct = 0
    i = 0
    for x, y in zip(x_test, y_test):
        output = predict(cnn, x)
        pred = np.argmax(output)
        truth = np.argmax(y)
        if pred == truth:
            correct += 1
        i += 1

    print(correct)
    print(correct / len(y_test))
