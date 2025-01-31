from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .dataloader import NaiveDataLoader
from .datasets import MNIST
from .inference import run_validation
from .loss import CELoss
from .model import CNNModel, load, save
from .optim import SGD, Adam


def solve_mnist(args):
    outdir = Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    mnist_data = MNIST("./mnist/")
    seed = torch.Generator().manual_seed(2024)

    train_dataset, val_dataset = torch.utils.data.random_split(
        mnist_data.data["train"], [0.9, 0.1], generator=seed
    )
    batch_size = 128
    train_loader = NaiveDataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = NaiveDataLoader(val_dataset, batch_size)
    test_loader = NaiveDataLoader(mnist_data.data["test"], batch_size)

    input_shape = mnist_data.input_shape
    output_shape = mnist_data.n_classes
    epoch = 20
    lr = 0.001

    loss_fn = CELoss()
    model = CNNModel(
        input_shape,
        output_shape,
        ks=[3],
        depths=[3],
        paddings=[0],
        fc_features=[128],
    )
    # optimizer = SGD(model.parameters(), lr, momentum=0.9)
    optimizer = Adam(model.parameters(), lr, betas=(0.9, 0.99))

    model_outp = f"{model.name}.mnist"
    initial_epoch = 0
    if args.preload is not None:
        checkpoint = outdir / f"{model_outp}.{args.preload}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError("Failed to find model file to preload.")
        state = load(checkpoint)
        initial_epoch = state.get("epoch") + 1
        model.load_state_dict(state.get("model_state_dict"))
        optimizer.load_state_dict(state.get("optimizer_state_dict"))

    for e in range(initial_epoch, epoch):
        tot_epoch_error = 0
        n_train = 0
        batch_iterator = tqdm(train_loader, desc=f"Processing epoch: {e:02d}")
        for batch in batch_iterator:
            batch_x = np.stack([train_loader.ds[idx][0] for idx in batch])
            batch_y = np.stack([train_loader.ds[idx][1] for idx in batch])
            model.train()
            optimizer.zero_grad()
            output = model(batch_x)
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
            batch_iterator.set_postfix({"loss": f"{ave_batch_loss:.6f}"})
        avg_epoch_error = tot_epoch_error / n_train
        accuracy = run_validation(val_loader, model)
        batch_iterator.write(
            f"ave_epoch_err={e+1}/{epoch} "
            f"avg_epoch_error={avg_epoch_error:.6f} accuracy={accuracy:.6f}"
        )
        model_fspath = outdir / f"{model_outp}.{e:02d}.pt"
        save(
            {
                "epoch": e,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_fspath,
        )

    accuracy = run_validation(test_loader, model)
    print(f"Test accuracy: {accuracy:.6f}")
