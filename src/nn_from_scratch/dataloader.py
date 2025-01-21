import numpy as np


class NaiveDataLoader:
    def __init__(self, ds, batch_size: int, shuffle: bool = False):
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(ds))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.ds), self.batch_size):
            j = min(i + self.batch_size, len(self.ds))
            batch_indices = self.indices[i:j]
            yield (
                np.stack(
                    [self.ds[idx][0].detach().numpy() for idx in batch_indices]
                ),
                np.stack(
                    [self.ds[idx][1].detach().numpy() for idx in batch_indices]
                ),
            )

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size + 1
