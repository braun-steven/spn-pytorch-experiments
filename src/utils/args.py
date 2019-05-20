import argparse
from src.utils.utils import ensure_dir
import json
import os


def init_argparser():
    """
    Define and parse commandline arguments.
    """
    # training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        metavar="N",
        help="input batch size for testing (default: same as --batch-size)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["verbose", "info", "warning", "error", "debug"],
        help="Log level",
    )
    parser.add_argument(
        "--result-dir",
        default="results",
        help="path to the result directory",
        metavar="DIR",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="N",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="Enable CUDA training"
    )

    parser.add_argument(
        "--cuda-device-id",
        nargs="+",
        type=int,
        default=[0],
        help="Cuda device ids. E.g. [0,1,2]. Use -1 for all GPUs available and -2 for cpu only.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debugging."
    )
    parser.add_argument(
        "--experiment-name", type=str, help="Set the experiment name", required=True
    )
    parser.add_argument("--net", type=str, help="Define network", required=True)
    parser.add_argument(
        "--n-gaussians",
        type=int,
        default=3,
        metavar="N",
        help="number of possible independence combinations of gaussians",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=4,
        metavar="S",
        help="Number of threads (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--tag",
        default="",
        type=str,
        help="Tag to identify runs in the result directory and tensorboard overviews",
    )
    parser.add_argument(
        "--resnet-arch",
        default="resnet18",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="Resnet architecture",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "iris-2d",
            "wine-2d",
            "diabetes",
            "audit",
            "banknotes",
            "ionosphere",
            "sonar",
            "wheat-2d",
            "synth-8-easy",
            "synth-64-easy",
            "synth-8-hard",
            "synth-64-hard",
        ],
    )
    parser.add_argument(
        "--force-overfit",
        action="store_true",
        default=False,
        help="Force overfitting (set num train samples to 1000)",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    return parser

    # args = parser.parse_args()
    # ensure_dir(args.result_dir)

    # if args.debug:
    #     args.epochs = 2

    # if args.n_digits > args.n_labels:
    #     raise Exception("Option --n-digits has to be <= --n-labels.")

    # return args


def load_args(result_dir):
    """Load the commandline arguments.

    """
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    with open(os.path.join(result_dir, "args.txt"), "r") as f:
        args.__dict__ = json.load(f)
    return args


def save_args(args):
    """Save the commandline arguments.

    Args:
        args: Commandline arguments.

    """
    with open(os.path.join(args.result_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
