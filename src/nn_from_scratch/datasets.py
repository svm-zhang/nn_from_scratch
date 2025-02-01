from collections import defaultdict

import numpy as np
import torch.nn.functional as F
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
        transforms = {"train": transform, "test": transform}
        target_transforms = {
            "train": transform_target,
            "test": transform_target,
        }
        self.data = {}
        for split in ["train", "test"]:
            self.data[split] = torchvision.datasets.MNIST(
                ds_dir,
                download=True,
                train=(split == "train"),
                transform=transforms[split],
                target_transform=target_transforms[split],
            )
        assert len(self.data["train"].classes) == len(
            self.data["test"].classes
        )
        self._n_classes = len(self.data["train"].classes)
        self._input_shape = self.data["train"][0][0].shape

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def input_shape(self):
        return self._input_shape


class FashionMNIST:
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
        transforms = {"train": transform, "test": transform}
        target_transforms = {
            "train": transform_target,
            "test": transform_target,
        }
        self.data = {}
        for split in ["train", "test"]:
            self.data[split] = torchvision.datasets.FashionMNIST(
                ds_dir,
                download=True,
                train=(split == "train"),
                transform=transforms[split],
                target_transform=target_transforms[split],
            )
        assert len(self.data["train"].classes) == len(
            self.data["test"].classes
        )
        self._n_classes = len(self.data["train"].classes)
        self._input_shape = self.data["train"][0][0].shape

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def input_shape(self):
        return self._input_shape


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


def subset(dataset, indices, n_per_label):
    counter = defaultdict(int)
    num_classes = dataset.targets.detach().numpy().max() + 1
    sub_xs = []
    sub_ys = []
    for idx in indices:
        x, y = (
            dataset.data[idx],
            dataset.targets[idx],
        )
        label = y.item()
        if counter[label] < n_per_label:
            x = x.float() / 255.0
            x = (
                torch.reshape(x, (1, x.shape[0], x.shape[1]))
                .detach()
                .numpy()
                .astype(np.float32)
            )
            y = F.one_hot(y, num_classes)
            sub_xs.append(x)
            sub_ys.append(y.detach().numpy())
            counter[label] += 1

    sub_x = np.array(sub_xs)
    sub_y = np.array(sub_ys)
    return sub_x, sub_y
