import numpy as np
import torch
from tqdm.auto import tqdm

from .activation import ReLU
from .dataloader import NaiveDataLoader
from .datasets import MNIST
from .inference import run_validation
from .layer import BatchNorm1D, Convolution, Dense, Reshape
from .loss import CELoss
from .model import build_cnn_model
from .optim import SGD
from .train import train


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

    input_shape = (1, 28, 28)
    output_shape = len(mnist_data.data.classes)
    epoch = 20
    lr = 0.001

    model = build_cnn_model(
        input_shape,
        output_shape,
        [3],
        [3],
        [0],
        [128],
    )
    optimizer = SGD(model.parameters(), lr, momentum=0.9)

    # network = [
    #     Convolution((1, 28, 28), kernel_size, n_kernel),
    #     ReLU(),
    #     Reshape((n_kernel, 26, 26), n_kernel * 26 * 26),
    #     Dense(n_kernel * 26 * 26, fc1),
    #     BatchNorm1D(fc1),
    #     ReLU(),
    #     Dense(fc1, 10),
    # ]

    loss_fn = CELoss()
    for e in range(epoch):
        tot_epoch_error = 0
        n_train = 0
        batch_iterator = tqdm(train_loader, desc=f"Processing epoch: {e:02d}")
        for batch in batch_iterator:
            batch_x = np.stack([train_loader.ds[idx][0] for idx in batch])
            batch_y = np.stack([train_loader.ds[idx][1] for idx in batch])
            optimizer.zero_grad()
            output = model.train(batch_x)
            batch_y_true_idx = batch_y.argmax(axis=1)
            loss = loss_fn.loss(batch_y_true_idx, output)
            grad = loss_fn.loss_prime(batch_y_true_idx, output)
            model.backward(grad)
            optimizer.step()
            assert isinstance(grad, np.ndarray)
            batch_tot_loss = np.sum(loss, axis=-1)
            tot_epoch_error += np.sum(batch_tot_loss)
            ave_batch_loss = np.mean(batch_tot_loss)
            n_train += len(batch_x)
            batch_iterator.set_postfix({"loss": f"{ave_batch_loss:.3f}"})
        avg_epoch_error = tot_epoch_error / n_train
        accuracy = run_validation(val_loader, model)
        batch_iterator.write(
            f"ave_epoch_err={e+1}/{epoch} "
            f"avg_epoch_error={avg_epoch_error:.3f} accuracy={accuracy:.3f}"
        )

    # train(train_loader, val_loader, model, optimizer, loss_fn, epoch)

    accuracy = run_validation(test_loader, model)
    print(f"Test accuracy: {accuracy:.3f}")
