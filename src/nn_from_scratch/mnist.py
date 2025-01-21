import torch

from .activation import ReLU
from .dataloader import NaiveDataLoader
from .datasets import MNIST
from .inference import run_validation
from .layer import BatchNorm1D, Convolution, Dense, Reshape
from .loss import CELoss
from .model import train


def solve_mnist():
    mnist_data = MNIST("./mnist/")
    seed = torch.Generator().manual_seed(2024)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        mnist_data.data, [0.85, 0.05, 0.1], generator=seed
    )
    batch_size = 128
    train_loader = NaiveDataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = NaiveDataLoader(val_dataset, batch_size)
    test_loader = NaiveDataLoader(test_dataset, batch_size)

    n_kernel = 3
    kernel_size = 3
    fc1 = 128

    network = [
        Convolution((1, 28, 28), kernel_size, n_kernel),
        ReLU(),
        Reshape((n_kernel, 26, 26), n_kernel * 26 * 26),
        Dense(n_kernel * 26 * 26, fc1),
        BatchNorm1D(fc1),
        ReLU(),
        Dense(fc1, 10),
    ]
    epoch = 30
    lr = 0.005

    loss_fn = CELoss()

    train(train_loader, val_loader, network, loss_fn, epoch, lr)

    accuracy = run_validation(test_loader, network)
    print(f"Test accuracy: {accuracy:.3f}")
