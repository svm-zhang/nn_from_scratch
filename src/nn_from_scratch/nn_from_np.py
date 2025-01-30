from .cli import parse_cmd
from .mnist import solve_mnist
from .torch_cnn import solve_mnist_with_torch
from .xor import solve_xor


def run():
    parser = parse_cmd()

    args = parser.parse_args()
    match args.dataset:
        case "xor":
            solve_xor()
        case "mnist":
            solve_mnist(args)
        case "mnist_torch":
            solve_mnist_with_torch()
        case _:
            pass
