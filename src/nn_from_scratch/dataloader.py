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

        for batch in np.array_split(self.indices, self.__len__()):
            yield batch

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size + 1
