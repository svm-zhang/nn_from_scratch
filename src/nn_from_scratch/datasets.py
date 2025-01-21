import numpy as np
import torch.utils
import torchvision
from torchvision.transforms import v2


class MNIST:
    def __init__(self, ds_dir: str):
        transform = v2.Compose(
            [v2.PILToTensor(), v2.ToDtype(torch.float, scale=True)]
        )
        transform_target = v2.Compose(
            [
                v2.Lambda(
                    lambda y: torch.zeros(10, dtype=torch.float).scatter_(
                        0, torch.tensor(y), value=1
                    )
                )
            ]
        )
        self.data = torchvision.datasets.MNIST(
            ds_dir,
            download=True,
            train=True,
            transform=transform,
            target_transform=transform_target,
        )


def extract_tensor(dataloader):
    xs = []
    ys = []
    for _, (x, y) in enumerate(dataloader):
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
