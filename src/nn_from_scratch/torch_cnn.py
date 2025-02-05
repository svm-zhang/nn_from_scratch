import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    test_loader = DataLoader(mnist_data.data["test"], batch_size)

    batch_size = 128

    epoches = 20
    input_shape = (1, 28, 28)
    kernel_size = 3
    depth = 3
    n_neurons = 128
    lr = 0.001
    loss_f = nn.CrossEntropyLoss()
    model = CNNModel(input_shape, kernel_size, depth, n_neurons)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    model_out = "test.pt"
    losses = []
    for epoch in range(epoches):
        epoch_loss = 0.0
        n_batch = 0
        batch_iterator = tqdm(
            train_loader, desc=f"Processing epoch: {epoch:02d}"
        )
        for x_train, y_train in batch_iterator:
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
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_out,
        )

    n_correct = 0
    n_test = 0
    with torch.inference_mode():
        batch_iterator = tqdm(test_loader, desc="Evaluating:")
        for x_test, y_test in batch_iterator:
            logits = model(x_test)
            pred = torch.softmax(logits, dim=-1).argmax(dim=-1).squeeze()
            truth = torch.argmax(y_test, dim=-1).squeeze()

            n_correct += torch.sum(pred == truth).item()
            n_test += y_test.shape[0]

    acc = n_correct / n_test

    print(f"{n_correct=} {n_test=} {acc=}")
