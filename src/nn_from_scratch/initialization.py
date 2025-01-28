import numpy as np


def he_normal_init(c_in: tuple[int, ...], c_out):
    stddev = np.sqrt(2.0 / np.prod(c_in))
    return np.random.normal(0, stddev, (c_out, *c_in))


def random_normal_init(c_in, c_out):
    return np.random.normal(0, 1, size=(c_out, *c_in))


def random_uniform_init(c_in, c_out):
    return np.random.rand(c_out, *c_in)  # w[j][i]
