"""
Plot results.
"""
import argparse
import os
import traceback
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import data_loader
from typing import Dict

from src.utils.utils import ensure_dir
from src.utils.args import load_args


def parse_args():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        description="SPFlow PyTorch Experiments plot script."
    )
    parser.add_argument(
        "--run", default="results", help="path to the result directory", metavar="DIR"
    )
    args = parser.parse_args()
    return args


def plot_batch_sizes():
    base_dir_acc = os.path.join(args.run)
    plot_dir = plot_base_dir
    ensure_dir(plot_dir)


def plot_accs_mnist_multilabel():
    base_dir_acc = os.path.join(args.run)
    plot_dir = plot_base_dir
    ensure_dir(plot_dir)
    ftype = "csv"

    fig, axes = plt.subplots(ncols=2)
    sns.despine()
    fig.set_figheight(5)
    fig.set_figwidth(15)

    # For each dataset and each suffix (configuration)
    for exp_name in exp_names:
        if exp_name == "spn-shallow":
            continue

        # Define file name
        csv_name = "{}/{}/mnist.{}".format(base_dir_acc, exp_name, ftype)
        # Load dataframes
        df = pd.read_csv(csv_name, sep=",", header=0)
        df.columns = ["epochs", "train_acc", "test_acc", "train_loss", "test_loss"]

        NEPOCHS = df.shape[0]
        x = range(NEPOCHS)

        alpha_train = 0.6
        alpha_test = 1.0
        linewidth_train = 2.0
        linewidth_test = 3.0

        # Plot accuracy
        axes[0].plot(
            x,
            df["train_acc"],
            label=exp_name.upper() + " Train",
            color=exp_colors[exp_name],
            alpha=alpha_train,
            linewidth=linewidth_train,
        )
        axes[0].plot(
            x,
            df["test_acc"],
            label=exp_name.upper() + " Test",
            color=exp_colors[exp_name],
            alpha=alpha_test,
            linewidth=linewidth_test,
        )

        # Plot loss
        axes[1].plot(
            x,
            df["train_loss"],
            label=exp_name.upper() + " Train",
            color=exp_colors[exp_name],
            alpha=alpha_train,
            linewidth=linewidth_train,
        )
        axes[1].plot(
            x,
            df["test_loss"],
            label=exp_name.upper() + " Test",
            color=exp_colors[exp_name],
            alpha=alpha_test,
            linewidth=linewidth_test,
        )

        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Accuracy")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")
        axes[0].set_ylim(70, 100)

        run_args = load_args(os.path.join(args.run, exp_name))
        n_labels = run_args.n_labels
        n_digits = run_args.n_digits
        batch_size = run_args.batch_size
        l2 = run_args.l2
        lr = run_args.lr

    # Titles
    title = "MNIST Multilabel ($N_{labels}=%s,N_{digits}=%s, bs=%s, l2=%s, lr=%s$)" % (
        n_labels,
        n_digits,
        batch_size,
        l2,
        lr,
    )
    fig.suptitle(title)
    # Shrink current axis's height by 10% on the bottom
    fontP = FontProperties()
    fontP.set_size("small")
    plt.legend(loc="upper right", fancybox=True, framealpha=0.5, prop=fontP)
    # fname = f"{n_labels:03}l_{n_digits:03}d.png"
    fname = f"{batch_size:03}.png"
    plt.savefig(os.path.join(plot_dir, fname), dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":

    # Apply seaborn default styling
    sns.set(context="notebook")
    sns.set_style("white")
    colors = sns.color_palette()
    dpi = 160

    args = parse_args()
    exp_names = sorted(os.listdir(args.run))
    os.path.isdir
    isdir = lambda path: os.path.isdir(os.path.join(args.run, path))
    exp_names = list(filter(isdir, exp_names))
    # If this result_dir has been visualized alread, exclude the plots dir
    if "plots" in exp_names:
        exp_names.remove("plots")

    exp_colors = {exp_names[i]: colors[i] for i in range(len(exp_names))}

    plot_base_dir = os.path.join(args.run, "plots")

    # Run plot generation
    plot_accs_mnist_multilabel()
