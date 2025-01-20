import numpy as np

from .layer import Dense
from .loss import Loss


def create_batch(x, y, batch_size):
    n_batch = int(np.ceil(len(x) / batch_size))
    batch_x = np.array_split(x, n_batch)
    batch_y = np.array_split(y, n_batch)
    return batch_x, batch_y


def train(
    x_train,
    y_train,
    network,
    loss_fn: Loss,
    batch_size: int,
    epoches: int = 100,
    lr: float = 0.1,
):
    for e in range(epoches):
        tot_epoch_error = 0
        # shuffle data and then make batch for each epoch
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        shuffled_x = x_train[indices]
        shuffled_y = y_train[indices]
        shuffled_batch_x, shuffled_batch_y = create_batch(
            shuffled_x, shuffled_y, batch_size
        )

        for _, (batch_x, batch_y) in enumerate(
            zip(shuffled_batch_x, shuffled_batch_y)
        ):
            output = batch_x
            for layer in network:
                # print(layer)
                output = layer.forward(output)
                # print(f"{output.shape=}")
                # print("-" * 50)
            # print(output.shape)
            # print(batch_y.shape)
            batch_y_true_idx = batch_y.argmax(axis=1)
            loss = loss_fn.loss(batch_y_true_idx, output)
            grad = loss_fn.loss_prime(batch_y_true_idx, output)
            for layer in network[::-1]:
                # print(layer)
                grad = layer.backward(grad, lr)
                # print("-" * 50)
            assert isinstance(grad, np.ndarray)
            # print(f"final: {grad.shape=}")
            batch_tot_loss = np.sum(loss, axis=-1)
            ave_batch_loss = np.mean(batch_tot_loss)
            tot_epoch_error += np.sum(batch_tot_loss)
            # print(f"{ave_batch_loss=}")
        avg_epoch_error = tot_epoch_error / (
            len(shuffled_batch_x) * batch_size
        )
        print(f"ave_epoch_err={e+1}/{epoches} {avg_epoch_error=}")
        print("-" * 100)

        #     for x, y in zip(batch_x, batch_y):
        #         output = x
        #         for layer in network:
        #             output = layer.forward(output)
        #
        #         my_loss = loss.loss(y.argmax(), output)
        #         tot_batch_error += my_loss
        #         grad = loss.loss_prime(y.argmax(), output)
        #
        #         for layer in network[::-1]:
        #             # print(layer)
        #             grad = layer.backward(grad, lr)
        #             # print(f"output gradient\n{grad=}\n{grad.shape}")
        #             # print("-" * 100)
        #
        #     avg_batch_error = tot_batch_error / len(batch_x)
        #     print(f"ave_err {n_batch=} {avg_batch_error=}")
        #     tot_epoch_error += tot_batch_error
        #     n_batch += 1
        # avg_epoch_error = tot_epoch_error / (n_batch * n_images_per_batch)
        # print(f"ave_epoch_err={e+1}/{epoches} {avg_epoch_error=}")
        # print("-" * 100)


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output
