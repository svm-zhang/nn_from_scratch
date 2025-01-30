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
    parser.add_argument(
        "--outdir",
        metavar="DIR",
        help="Specify path to output directory.",
    )
    parser.add_argument(
        "--preload",
        metavar="STR",
        help="Specify checkpoint to load.",
    )

    return parser
