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
                    label=exp_name.upper() + " Train",
                    color=exp_colors[exp_name],
                )
                plt.plot(
                    x,
                    df["test_acc"],
                    label=exp_name.upper() + " Test",
                    color=exp_colors[exp_name],
                    alpha=0.66,
                )

            # Titles
            title = "{}".format(ds_name).capitalize()
            plt.title(title)

            # Y axis
            plt.ylabel("Loss")
            plt.ylabel("Accuracy")
            plt.ylim((40, 105))

            # X axis
            plt.xlabel("Epochs")
            plt.legend(loc="lower right")
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


def plot_fewshot_results():
    base_dir_acc = os.path.join(args.run)
    plot_dir = os.path.join(plot_base_dir, "mlp-spn")
    ensure_dir(plot_dir)

    # Define file name
    fname_mlp = "{}/{}/mnist.csv".format(base_dir_acc, "mlp")
    fname_spn = "{}/{}/mnist.csv".format(base_dir_acc, "spn")

    # Load dataframes
    df_mlp = pd.read_csv(fname_mlp, sep=",", header=0)
    df_spn = pd.read_csv(fname_spn, sep=",", header=0)

    # Fix columns

    # Plot each df in its own subplot
    fig = plt.figure()
    # fig.set_figheight(10)
    # fig.set_figwidth(15)

    column_names = ["pct", "train_acc", "test_acc", "train_loss", "test_loss"]
    df_mlp.columns = column_names
    df_spn.columns = column_names

    # Plot accuracy
    mask_spn = df_spn["pct"] < 2.5
    mask_mlp = df_mlp["pct"] < 2.5

    # Train
    plt.plot(
        df_mlp["pct"][mask_mlp],
        df_mlp["train_acc"][mask_mlp],
        label="Train MLP",
        color=exp_colors["mlp"],
        alpha=1.0,
    )
    plt.plot(
        df_spn["pct"][mask_spn],
        df_spn["train_acc"][mask_spn],
        label="Train SPN",
        color=exp_colors["spn"],
        alpha=1.0,
    )

    # Test
    plt.plot(
        df_mlp["pct"][mask_mlp],
        df_mlp["test_acc"][mask_mlp],
        label="Test MLP",
        color=exp_colors["mlp"],
        alpha=0.66,
    )
    plt.plot(
        df_spn["pct"][mask_spn],
        df_spn["test_acc"][mask_spn],
        label="Test SPN",
        color=exp_colors["spn"],
        alpha=0.66,
    )

    plt.title("Accuracy vs Pct of Train Data (MNIST)")
    plt.xlabel("Train Data Percentage")
    plt.ylabel("Accuracy (%)")
    plt.ylim((20, 105))
    plt.xlim((0, 2.6))

    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, "result.png"), dpi=dpi)


def plot_accs_multiple_percentages():
    base_dir_acc = os.path.join(args.run)
    plot_dir = os.path.join(plot_base_dir, "mlp-spn")
    ensure_dir(plot_dir)

    min_percentage = 0.05  # Equals 3 images per class
    max_percentage = 2.5
    percentage_step_size = 0.1

    # Iterate over percentages in  steps
    rng = np.arange(
        min_percentage, max_percentage + min_percentage, percentage_step_size
    )
    ftype = "csv"

    # For each dataset and each suffix (configuration)
    for p in rng:
        fname = "mnist-p={0:.2f}".format(p)
        # Plot each df in its own subplot
        fig = plt.figure()

        for exp_name in exp_names:

            # Define file name
            csv_name = "{}/{}/{}.{}".format(base_dir_acc, exp_name, fname, ftype)
            # Load dataframes
            df = pd.read_csv(csv_name, sep=",", header=0)
            df.columns = ["epochs", "train_acc", "test_acc", "train_loss", "test_loss"]

            NEPOCHS = df.shape[0]
            x = range(NEPOCHS)
            plt.plot(
                x,
                df["train_acc"],
                label=exp_name.upper() + " Train",
                color=exp_colors[exp_name],
            )
            plt.plot(
                x,
                df["test_acc"],
                label=exp_name.upper() + " Test",
                color=exp_colors[exp_name],
                alpha=0.66,
            )

        # Titles
        title = "MLP vs SPN (MNIST: p={:.2f} )".format(p)
        plt.title(title)

        # Y axis
        plt.ylabel("Accuracy")
        plt.ylim((10, 105))

        # X axis
        plt.xlabel("Epochs")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(plot_dir, "{}.png".format(fname)), dpi=dpi)


def plot_auc_mnist_multilabel():
    base_dir_acc = os.path.join(args.run)
    plot_dir = os.path.join(plot_base_dir, "mlp-spn")
    ensure_dir(plot_dir)
    ftype = "csv"

    fig, axes = plt.subplots(ncols=2)
    fig.set_figheight(5)
    fig.set_figwidth(15)

    # For each dataset and each suffix (configuration)
    for exp_name in exp_names:

        # Define file name
        csv_name = "{}/{}/mnist.{}".format(base_dir_acc, exp_name, ftype)
        # Load dataframes
        df = pd.read_csv(csv_name, sep=",", header=0)
        df.columns = ["epochs", "train_auc", "test_auc", "train_loss", "test_loss"]

        NEPOCHS = df.shape[0]
        x = range(NEPOCHS)

        # Plot accuracy
        axes[0].plot(
            x,
            df["train_auc"],
            label=exp_name.upper() + " Train",
            color=exp_colors[exp_name],
        )
        axes[0].plot(
            x,
            df["test_auc"],
            label=exp_name.upper() + " Test",
            color=exp_colors[exp_name],
            alpha=0.66,
        )

        # Plot loss
        axes[1].plot(
            x,
            df["train_loss"],
            label=exp_name.upper() + " Train",
            color=exp_colors[exp_name],
        )
        axes[1].plot(
            x,
            df["test_loss"],
            label=exp_name.upper() + " Train",
            color=exp_colors[exp_name],
            alpha=0.66,
        )

        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("AUC")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")
        axes[0].set_ylim(0.3, 1.05)

    n_labels = args.run.split("nlabels=")[1].replace("/", "")

    # Titles
    title = "MLP vs SPN ($N_{labels}=%s$)" % n_labels
    plt.title(title)

    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, "result.png"), dpi=dpi)


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
        # axes[0].set_ylim(65, 100)

    run_args = load_args(args.run)
    n_labels = run_args.n_labels
    n_digits = run_args.n_digits

    # Titles
    title = "MNIST Multilabel ($N_{labels}=%s,N_{digits}=%s$)" % (n_labels, n_digits)
    fig.suptitle(title)
    # Shrink current axis's height by 10% on the bottom
    fontP = FontProperties()
    fontP.set_size("small")
    plt.legend(loc="upper right", fancybox=True, framealpha=0.5, prop=fontP)
    plt.savefig(os.path.join(plot_dir, "result.png"), dpi=dpi, bbox_inches="tight")


def plot_n_gaussians():
    base_dir_acc = os.path.join(args.run)
    plot_dir = os.path.join(plot_base_dir, "spn")
    ensure_dir(plot_dir)
    ftype = "csv"
    # Plot each df in its own subplot
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    # For each dataset and each suffix (configuration)
    for i in range(2, 11):
        exp_name = "n-gaussians=%s" % i

        # Define file name
        csv_name = os.path.join(base_dir_acc, exp_name, "mnist.csv")
        # Load dataframes
        df = pd.read_csv(csv_name, sep=",", header=0)
        df.columns = ["epochs", "train_acc", "test_acc", "train_loss", "test_loss"]

        NEPOCHS = df.shape[0]
        x = range(NEPOCHS)

        # Axes:
        # 0,0 = Train Loss
        # 1,0 = Test Loss
        # 0,1 = Train Accuracy
        # 1,1 = Test Accuracy

        # Set mins/maxs
        axes[0, 1].set_ylim(50, 90)
        axes[1, 1].set_ylim(50, 90)
        axes[0, 0].set_ylim(1, 4)
        axes[1, 0].set_ylim(1, 4)

        # Plot loss
        axes[0, 0].plot(x, df["train_loss"], label="$N=%s$" % i)
        axes[1, 0].plot(x, df["test_loss"], label="$N=%s$" % i)

        # Plot accuracy
        axes[0, 1].plot(x, df["train_acc"], label="$N=%s$" % i)
        axes[1, 1].plot(x, df["test_acc"], label="$N=%s$" % i)

        # Y axis
        axes[0, 0].set_ylabel("Train Loss")
        axes[1, 0].set_ylabel("Test Loss")
        axes[0, 1].set_ylabel("Train Accuracy")
        axes[1, 1].set_ylabel("Test Accuracy")

        # X axis
        axes[1, 0].set_xlabel("Epochs")
        axes[1, 1].set_xlabel("Epochs")
        title = "MNIST: Loss/Accuracy over Epochs ($N_{labels}=5$)"

    fig.suptitle(title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, "result.png"), dpi=240)


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
