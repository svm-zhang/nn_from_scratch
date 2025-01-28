from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
from tqdm.auto import tqdm

from .dataloader import NaiveDataLoader
from .datasets import MNIST


class CNNModel(nn.Module):
    def __init__(self, input_shape, kernel_size, depth, n_neurons):
        super(CNNModel, self).__init__()
        in_channels, in_height, in_width = input_shape

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, depth, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                depth
                * (in_height - kernel_size + 1)
                * (in_width - kernel_size + 1),
                n_neurons,
            ),
            nn.ReLU(),
            nn.Linear(n_neurons, 10),
        )

    def forward(self, input):
        return self.network(input)


def solve_mnist_with_torch():
    mnist_data = MNIST("./mnist/")
    seed = torch.Generator().manual_seed(2024)
    train_dataset, val_dataset = torch.utils.data.random_split(
        mnist_data.data["train"], [0.9, 0.1], generator=seed
    )
    batch_size = 128
    train_loader = NaiveDataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = NaiveDataLoader(val_dataset, batch_size)
    test_loader = NaiveDataLoader(mnist_data.data["test"], batch_size)

    batch_size = 128

    epoches = 50
    input_shape = (1, 28, 28)
    kernel_size = 3
    depth = 3
    n_neurons = 128
    lr = 0.001
    loss_f = nn.CrossEntropyLoss()
    model = CNNModel(input_shape, kernel_size, depth, n_neurons)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epoches):
        epoch_loss = 0.0
        n_batch = 0
        for x_train, y_train in train_loader:
            model.train()
            optimizer.zero_grad()
            logits = model(x_train)
            loss = loss_f(logits, y_train)

            loss.backward()
            optimizer.step()

            epoch_loss += loss
            n_batch += 1
        avg_epoch_loss = epoch_loss / n_batch
        losses.append(avg_epoch_loss)
        print(f"{epoch=} {avg_epoch_loss=}")

    n_correct = 0
    with torch.inference_mode():
        for x_test, y_test in test_loader:
            logits = model(x_test)
            pred = torch.softmax(logits, dim=-1).argmax(dim=-1).squeeze()
            y_test = y_test.squeeze()

            if torch.equal(pred, y_test):
                n_correct += 1
            else:
                print(f"{pred=} {y_test=}")

    acc = n_correct / len(test_loader)
    print(f"{acc=}")
