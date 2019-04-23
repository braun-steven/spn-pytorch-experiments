import argparse
from src.utils.utils import ensure_dir


def parse_args():
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
        "--n-labels",
        type=int,
        default=2,
        metavar="N",
        help="Number of labels for artificial multilabel mnist task",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="Enable CUDA training"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debugging."
    )
    parser.add_argument(
        "--experiment-name", type=str, help="Set the experiment name", required=True
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=["spn", "mlp", "mlplist"],
        help="Define network",
        required=True,
    )
    parser.add_argument(
        "--n-gaussians",
        type=int,
        default=2,
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
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    ensure_dir(args.result_dir)

    # If no test batch size was given, use 5 times the train batch size
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size * 5

    if args.debug:
        args.epochs = 2

    return args
