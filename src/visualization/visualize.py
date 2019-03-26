"""
Plot results.
"""
import argparse
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import data_loader
from typing import Dict

from src.utils.utils import ensure_dir

# Apply seaborn default styling
sns.set()
sns.set_palette("deep")
colors = sns.color_palette()
exp_names = ["mlp", "spn"]
exp_colors = {"mlp": colors[0], "spn": colors[1]}


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


def plot_accuracies():
    """Plot the accuracies of train/test for different models"""
    base_dir_acc = os.path.join(args.run)
    plot_dir = os.path.join(plot_base_dir, "mlp-spn")
    ensure_dir(plot_dir)

    # For each dataset and each suffix (configuration)
    for ds_name in dataset_names:
        try:
            # Plot each df in its own subplot
            fig = plt.figure()
            fig.set_figheight(10)
            fig.set_figwidth(15)

            for exp_name in exp_names:

                # Define file name
                fname = "{}/{}/{}.{}".format(base_dir_acc, exp_name, ds_name, ftype)
                # Load dataframes
                df = pd.read_csv(fname, sep=",", header=0)
                df.columns = ["train_acc", "test_acc", "train_loss", "test_loss"]

                NEPOCHS = df.shape[0]
                x = range(NEPOCHS)
                plt.plot(
                    x,
                    df["train_acc"],
                    label=exp_name + " train",
                    color=exp_colors[exp_name],
                )
                plt.plot(
                    x,
                    df["test_acc"],
                    label=exp_name + " test",
                    color=exp_colors[exp_name],
                    alpha=0.66,
                )

            # Titles
            title = "{}: Accuracy over Epochs".format(ds_name)
            plt.title(title)

            # Y axis
            plt.ylabel("Loss")
            plt.ylabel("Accuracy")

            # X axis
            # plt.set_xlabel("Epochs")
            plt.xlabel("Epochs")
            plt.legend()
            plt.savefig(os.path.join(plot_dir, "{}.png".format(ds_name)), dpi=dpi)
        except Exception as e:
            print("Error in Dataset:", ds_name)
            print(e)
            print(repr(e))
            traceback.print_exc()


def plot_epoch_loss_acc():
    base_dir_acc = os.path.join(args.run)
    plot_dir = os.path.join(plot_base_dir, "mlp-spn")
    ensure_dir(plot_dir)

    # For each dataset and each suffix (configuration)
    for ds_name in dataset_names:
        try:
            for exp_name in exp_names:

                # Define file name
                acc_train_fname = "{}/{}/{}.{}".format(
                    base_dir_acc, exp_name, ds_name, ftype
                )
                # Load dataframes
                df = pd.read_csv(acc_train_fname, sep=",", header=0)

                # Fix columns

                # Plot each df in its own subplot
                fig, axes = plt.subplots(nrows=2, ncols=1)
                fig.set_figheight(10)
                fig.set_figwidth(15)

                column_names = ["train_acc", "test_acc", "train_loss", "test_loss"]
                df.columns = column_names

                # Plot accuracy
                axes[1].plot(range(NEPOCHS), df[column_names[0]].values[:NEPOCHS])
                axes[1].plot(range(NEPOCHS), df[column_names[1]].values[:NEPOCHS])

                # Plot loss
                axes[0].plot(range(NEPOCHS), df[column_names[2]].values[:NEPOCHS])
                axes[0].plot(range(NEPOCHS), df[column_names[3]].values[:NEPOCHS])

                # Titles
                axes[0].set_title("Loss")
                axes[1].set_title("Accuracy")

                # Y axis
                axes[0].set_ylabel("Loss")
                axes[1].set_ylabel("Accuracy")

                # X axis
                # axes[0].set_xlabel("Epochs")
                axes[1].set_xlabel("Epochs")
                title = "{} - {}:  Loss/Accuracy over Epochs".format(ds_name, exp_name)
                fig.suptitle(title)
                plt.legend(labels=["Train", "Test"])
                plt.savefig(
                    os.path.join(plot_dir, "{}-{}.png".format(ds_name, exp_name)),
                    dpi=dpi,
                )
        except Exception as e:
            print("Error in Dataset:", ds_name)
            print(e)
            print(repr(e))
            traceback.print_exc()


if __name__ == "__main__":
    dpi = 120
    NEPOCHS = 300
    ftype = "csv"
    dataset_names = [
        "iris-2d",
        "wine-2d",
        "diabetes",
        "audit",
        "banknotes",
        "ionosphere",
        "sonar",
        "wheat-2d",
        "synth-8-easy",
        "synth-8-hard",
        "synth-64-easy",
        "synth-64-hard",
    ]
    dataset_size = {}
    for name, loader in data_loader.load_dataset_map().items():
        X, y = loader()
        n = X.shape[0]
        dataset_size[name] = n

    args = parse_args()
    plot_base_dir = os.path.join(args.run, "plots")

    # Run plot generation
    # plot_epoch_loss_acc()
    plot_accuracies()
