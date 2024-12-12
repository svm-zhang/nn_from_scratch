from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import v2


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
            x = torch.reshape(x, (1, x.shape[0], x.shape[1]))
            sub_xs.append(x)
            # y here is not a one-hot encoding, rather it is class index
            sub_ys.append(y)
            counter[label] += 1

    sub_x = torch.stack(sub_xs, 0)
    sub_y = torch.stack(sub_ys)
    return torch.utils.data.TensorDataset(sub_x, sub_y)


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
            nn.ReLU(),
        )

    def forward(self, input):
        return self.network(input)


def solve_mnist_with_torch():
    mnist_data = mnist()
    seed = torch.Generator().manual_seed(2024)
    train_dataset, test_dataset = torch.utils.data.random_split(
        mnist_data, [0.7, 0.3], generator=seed
    )

    n_train_per_label = 2048
    n_test_per_label = 128
    batch_size = 128
    train_subset = subset(mnist_data, train_dataset.indices, n_train_per_label)
    test_subset = subset(mnist_data, test_dataset.indices, n_test_per_label)
    train_subset_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True
    )
    test_subset_loader = torch.utils.data.DataLoader(
        test_subset, shuffle=False
    )

    epoches = 50
    input_shape = (1, 28, 28)
    kernel_size = 3
    depth = 8
    n_neurons = 128
    lr = 0.05
    loss_f = nn.CrossEntropyLoss()
    model = CNNModel(input_shape, kernel_size, depth, n_neurons)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epoches):
        epoch_loss = 0.0
        n_batch = 0
        for x_train, y_train in train_subset_loader:
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
        for x_test, y_test in test_subset_loader:
            logits = model(x_test)
            pred = torch.softmax(logits, dim=-1).argmax(dim=-1).squeeze()
            y_test = y_test.squeeze()

            if torch.equal(pred, y_test):
                n_correct += 1
            else:
                print(f"{pred=} {y_test=}")

    acc = n_correct / len(test_subset_loader)
    print(f"{acc=}")
