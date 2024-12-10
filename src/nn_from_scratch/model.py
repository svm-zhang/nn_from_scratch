from .loss import Loss


def train(
    x_train, y_train, network, loss: Loss, epoches: int = 100, lr: float = 0.1
):
    for e in range(epoches):
        error = 0
        for x, y in zip(x_train, y_train):
            output = predict(network, x)

            error += loss.loss(y, output)

            grad = loss.loss_prime(y, output)

            for layer in network[::-1]:
                grad = layer.backward(grad, lr)

        error /= len(x_train)
        print(f"{e+1}/{epoches} {error=}")


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output
