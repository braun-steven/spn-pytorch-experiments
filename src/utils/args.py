import argparse


def parse_args():
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
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
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
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debugging."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=["spn", "mlp", "mlplist"],
        help="Define network",
        required=True,
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
    return args
