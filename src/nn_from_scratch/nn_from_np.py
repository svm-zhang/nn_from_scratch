from .cli import parse_cmd
from .mnist import solve_mnist
from .xor import solve_xor


def run():
    parser = parse_cmd()

    args = parser.parse_args()
    match args.dataset:
        case "xor":
            solve_xor()
        case "mnist":
            solve_mnist()
        case _:
            pass
