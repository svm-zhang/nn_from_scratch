import argparse


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        metavar="STR",
        required=True,
        choices=["mnist", "xor", "mnist_torch"],
        help="Specify the dataset to train and test on.",
    )

    return parser
