import numpy as np
from tqdm.auto import tqdm

from .inference import run_validation
from .loss import Loss


def train(
    train_loader,
    val_loader,
    model,
    optimizer,
    loss_fn: Loss,
    epoches: int = 100,
):
    for e in range(epoches):
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
            f"ave_epoch_err={e+1}/{epoches} "
            f"avg_epoch_error={avg_epoch_error:.3f} accuracy={accuracy:.3f}"
        )
