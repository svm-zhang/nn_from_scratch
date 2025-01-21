import numpy as np

from .layer import Softmax


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def run_validation(loader, network):
    softmax = Softmax()
    n_correct, n_test = 0, 0
    for batch in loader:
        batch_x = np.stack([loader.ds[idx][0] for idx in batch])
        batch_y = np.stack([loader.ds[idx][1] for idx in batch])
        output = predict(network, batch_x)
        probs = softmax.forward(output)
        preds = np.argmax(probs, axis=-1)
        y_true = np.argmax(batch_y, axis=1)
        n_correct += (preds == y_true).sum()
        n_test += len(batch_x)
    return n_correct / n_test
