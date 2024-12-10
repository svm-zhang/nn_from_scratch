import numpy as np
import torch.utils
import torchvision
from torchvision.transforms import v2


def mnist():
    transform = v2.Compose([v2.PILToTensor()])
    mnist_data = torchvision.datasets.MNIST(
        "./mnist/", download=True, train=True, transform=transform
    )
    return mnist_data


def extract_tensor(dataloader):
    xs = []
    ys = []
    for i, (x, y) in enumerate(dataloader):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).detach().numpy(), torch.cat(ys).detach().numpy()


def preprocess(x, y, size):
    zero_index = np.where(y == 0)[0][:size]
    one_index = np.where(y == 1)[0][:size]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np.eye(2)[y]
    y = y.reshape(len(y), 2, 1)
    return x, y
