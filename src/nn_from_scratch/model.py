import numpy as np
from tqdm.auto import tqdm

from .loss import Loss

# TODO::
# 1. Add validation during training and record accuracy
# 2. Use tqdm [DONE]


def create_batch(x, y, batch_size):
    n_batch = int(np.ceil(len(x) / batch_size))
    batch_x = np.array_split(x, n_batch)
    batch_y = np.array_split(y, n_batch)
    return batch_x, batch_y


# def train(
#     x_train,
#     y_train,
#     network,
#     loss_fn: Loss,
#     batch_size: int,
#     epoches: int = 100,
#     lr: float = 0.1,
# ):
def train(
    train_loader,
    test_loader,
    network,
    loss_fn: Loss,
    batch_size: int,
    epoches: int = 100,
    lr: float = 0.1,
):
    for e in range(epoches):
        tot_epoch_error = 0
        n_train = 0
        batch_iterator = tqdm(train_loader, desc=f"Processing epoch: {e:02d}")
        for batch_x, batch_y in batch_iterator:
            output = batch_x
            for layer in network:
                output = layer.forward(output)
            batch_y_true_idx = batch_y.argmax(axis=1)
            loss = loss_fn.loss(batch_y_true_idx, output)
            grad = loss_fn.loss_prime(batch_y_true_idx, output)
            for layer in network[::-1]:
                grad = layer.backward(grad, lr)
            assert isinstance(grad, np.ndarray)
            batch_tot_loss = np.sum(loss, axis=-1)
            tot_epoch_error += np.sum(batch_tot_loss)
            ave_batch_loss = np.mean(batch_tot_loss)
            n_train += len(batch_x)
            batch_iterator.set_postfix({"loss": f"{ave_batch_loss:6.3f}"})
        avg_epoch_error = tot_epoch_error / n_train
        print(f"ave_epoch_err={e+1}/{epoches} {avg_epoch_error=}")


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output
