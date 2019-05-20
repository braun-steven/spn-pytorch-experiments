from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import os

from tqdm import tqdm
from matplotlib import pyplot as plt

from observations.util import maybe_download_and_extract
from observations.mnist import mnist

from typing import Dict, Callable
from torchvision import transforms, datasets

import numpy as np
import logging

from sklearn.datasets import load_iris, load_wine, make_classification

BASE_DIR = "data/raw/"

logger = logging.getLogger(__name__)


def load_audit():
    """Load the audit dataset"""
    data = np.loadtxt(BASE_DIR + "audit_risk.csv", delimiter=",", comments="#")
    X = data[:, 0:-2]
    y = data[:, -1]
    return X, y


def load_diabetes():
    """
    The Pima Indians Diabetes Dataset involves predicting the onset of diabetes within 5 years in Pima Indians given medical details.

    It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 768 observations with 8 input variables and 1 output variable. Missing values are believed to be encoded with zero values. The variable names are as follows:

    Number of times pregnant.
    Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
    Diastolic blood pressure (mm Hg).
    Triceps skinfold thickness (mm).
    2-Hour serum insulin (mu U/ml).
    Body mass index (weight in kg/(height in m)^2).
    Diabetes pedigree function.
    Age (years).
    Class variable (0 or 1).
    The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 65%. Top results achieve a classification accuracy of approximately 77%.
    """
    data = np.loadtxt(BASE_DIR + "diabetes.csv", delimiter=",")
    X = data[:, 0:-1]
    y = data[:, -1]
    return X, y


def load_sonar():
    """
    The Sonar Dataset involves the prediction of whether or not an object is a mine or a rock given the strength of sonar returns at different angles.

    It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 208 observations with 60 input variables and 1 output variable. The variable names are as follows:

    Sonar returns at different angles
    …
    Class (M for mine and R for rock)
    The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 53%. Top results achieve a classification accuracy of approximately 88%.
    """
    data = np.loadtxt(BASE_DIR + "sonar.csv", delimiter=",")
    X = data[:, 0:-1]
    y = data[:, -1]
    return X, y


def load_wheat_2d():
    """
    The Wheat Seeds Dataset involves the prediction of species given measurements of seeds from different varieties of wheat.

    It is a binary (2-class) classification problem. The number of observations for each class is balanced. There are 210 observations with 7 input variables and 1 output variable. The variable names are as follows:

    Area.
    Perimeter.
    Compactness
    Length of kernel.
    Width of kernel.
    Asymmetry coefficient.
    Length of kernel groove.
    Class (1, 2, 3).
    The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 28%.

    """
    data = np.loadtxt(BASE_DIR + "wheat.csv")
    X = data[:, 0:-1]
    y = data[:, -1]

    c0 = 1
    c1 = 3
    mask = (y == c0) | (y == c1)
    X = X[mask]
    y = y[mask]
    y[y == c0] = 0
    y[y == c1] = 1

    return X, y


def load_ionosphere():
    """
    The Ionosphere Dataset requires the prediction of structure in the atmosphere given radar returns targeting free electrons in the ionosphere.

It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 351 observations with 34 input variables and 1 output variable. The variable names are as follows:

17 pairs of radar return data.
…
Class (g for good and b for bad).
The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 64%. Top results achieve a classification accuracy of approximately 94%."""
    data = np.loadtxt(BASE_DIR + "ionosphere.csv", delimiter=",")
    X = data[:, 0:-1]
    y = data[:, -1]
    return X, y


def load_banknotes():
    """
    The Banknote Dataset involves predicting whether a given banknote is authentic given a number of measures taken from a photograph.

    It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 1,372 observations with 4 input variables and 1 output variable. The variable names are as follows:

    Variance of Wavelet Transformed image (continuous).
    Skewness of Wavelet Transformed image (continuous).
    Kurtosis of Wavelet Transformed image (continuous).
    Entropy of image (continuous).
    Class (0 for authentic, 1 for inauthentic).
    The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 50%.
    """
    data = np.loadtxt(BASE_DIR + "banknotes.csv", delimiter=",")
    X = data[:, 0:-1]
    y = data[:, -1]
    return X, y


def load_iris_3d():
    return load_iris(return_X_y=True)


def load_iris_2d():
    X, y = load_iris(return_X_y=True)
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]
    y[y == 1] = 0
    y[y == 2] = 1
    return X, y


def load_wine_3d():
    X, y = load_wine(return_X_y=True)
    return X, y


def load_wine_2d():
    X, y = load_wine(return_X_y=True)
    mask = (y == 0) | (y == 2)
    X = X[mask]
    y = y[mask]
    y[y == 0] = 0
    y[y == 2] = 1
    return X, y


def make_load_synth(
    n_features,
    n_informative,
    n_redundant,
    n_clusters_per_class,
    class_sep,
    n_datapoints=3000,
) -> Callable:
    """
    Create a synthetic dataset.

    Args:
        n_features (int): Number of input features.
        n_informative (int): Number of informative input features.
        n_redundant (int): Number of redundant input features.
        n_clusters_per_class (int): Number of cluster per class.
        class_sep (float): Class separation factor.
        n_datapoints (int): Number of datapoints (default: 3000).

    Returns:
        Callable: Dataset loader which returns (X, y).
    """

    def load_synth():
        X, y = make_classification(
            n_samples=n_datapoints,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=2,
            n_clusters_per_class=2,
            class_sep=class_sep,
            random_state=42,
        )
        return X, y

    return load_synth


def load_dataset_map() -> Dict[str, Callable]:
    """
    Get a map of datasets.

    Returns:
        Dict[str, Callable]: Key: Dataset name, Value: loader function which returns (X, y).
    """
    dss = {
        "iris-2d": load_iris_2d,
        "wine-2d": load_wine_2d,
        "diabetes": load_diabetes,
        "audit": load_audit,
        "banknotes": load_banknotes,
        "ionosphere": load_ionosphere,
        "sonar": load_sonar,
        "wheat-2d": load_wheat_2d,
    }
    for n_feats in [8, 64, 128, 256, 512, 1024, 2048]:
        dss["synth-" + str(n_feats) + "-easy"] = make_load_synth(
            n_features=n_feats,
            n_informative=n_feats,
            n_redundant=0,
            n_clusters_per_class=2,
            class_sep=0.5,
        )
        dss["synth-" + str(n_feats) + "-hard"] = make_load_synth(
            n_features=n_feats,
            n_informative=int(n_feats / 4),
            n_redundant=int(n_feats / 2),
            n_clusters_per_class=2,
            class_sep=0.01,
        )
    return dss


def get_mnist_subset(train_bs, test_bs, use_cuda=False, p=100):
    """
    Get MNIST dataset with a certain percentage of samples per class in the train set and the 
    full test set.

    Args:
        train_bs: Train batch size.
        test_bs: Test batch size.
        use_cuda: Flag to enable cuda.
        p: Percentage.
    """

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    mnist_train = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    max_samples_per_class = p / 100.0 * 60000 / 10

    # Select only p % of each class of the train set
    counter = [0] * 10
    data = []
    targets = []

    for sample in mnist_train:
        target = sample[1]

        if counter[target] < max_samples_per_class:
            counter[target] += 1
            data.append(sample[0])
            targets.append(target)

    mnist_train = torch.utils.data.TensorDataset(
        torch.stack(data), torch.Tensor(targets)
    )

    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=train_bs, shuffle=True, **kwargs
    )

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=test_bs,
        **kwargs,
    )

    return train_loader, test_loader


def get_cifar100_loader(
    use_cuda, args, train_sampler=None, test_sampler=None, size=(28, 28)
):
    """
    Get the CIFAR100 pytorch data loader.
    
    Args:
        use_cuda: Use cuda flag.
        args: Command line arguments.
        sampler: Dataset sampler.

    """

    kwargs = {"num_workers": 1, "pin_memory": False} if use_cuda else {}

    if args.debug:
        train_sampler = torch.utils.data.SubsetRandomSampler(
            indices=np.random.randint(0, 60000, (args.batch_size))
        )
        test_sampler = torch.utils.data.SubsetRandomSampler(
            indices=np.random.randint(0, 10000, (args.batch_size))
        )

    transformer = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100("../data", train=True, download=True, transform=transformer),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **kwargs,
    )

    # If test batch size is not set, use all of the test data
    if args.test_batch_size is None:
        args.test_batch_size = 10000

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100("../data", train=False, transform=transformer),
        batch_size=args.test_batch_size,
        shuffle=test_sampler is None,
        sampler=test_sampler,
        **kwargs,
    )
    return train_loader, test_loader


def get_mnist_loaders(
    use_cuda, args, train_sampler=None, test_sampler=None, size=(28, 28)
):
    """
    Get the MNIST pytorch data loader.
    
    Args:
        use_cuda: Use cuda flag.
        args: Command line arguments.
        sampler: Dataset sampler.

    """

    kwargs = {"num_workers": 1, "pin_memory": False} if use_cuda else {}

    if args.debug:
        train_sampler = torch.utils.data.SubsetRandomSampler(
            indices=np.random.randint(0, 60000, (args.batch_size))
        )
        test_sampler = torch.utils.data.SubsetRandomSampler(
            indices=np.random.randint(0, 10000, (args.batch_size))
        )

    transformer = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, download=True, transform=transformer),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **kwargs,
    )

    # If test batch size is not set, use all of the test data
    if args.test_batch_size is None:
        args.test_batch_size = 10000

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, transform=transformer),
        batch_size=args.test_batch_size,
        shuffle=test_sampler is None,
        sampler=test_sampler,
        **kwargs,
    )
    return train_loader, test_loader


def get_multilabel_mnist_loader(n_labels, use_cuda, args, size=(28, 28)):
    """
    Get multilabel MNIST pytorch data loader.
    
    Args:
        n_labels (int): Number of labels
        use_cuda: Use cuda flag.
        args: Command line arguments.
        sampler: Dataset sampler.
    """

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    transformer = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Get datasets
    mnist_train = datasets.MNIST(
        "../data", train=True, download=True, transform=transformer
    )
    mnist_test = datasets.MNIST("../data", train=False, transform=transformer)
    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=1, shuffle=True, **kwargs
    )

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=1, shuffle=True, **kwargs
    )

    def make_multilabel_loader(loader, batch_size: int):
        loaders = [
            torch.utils.data.DataLoader(
                mnist_train, batch_size=1, shuffle=True, **kwargs
            )
            for _ in range(n_labels)
        ]

        tensors_data = []
        tensors_targets = []
        for idx, entry in enumerate(zip(*loaders)):
            data = []
            targets = []
            for i in range(n_labels):
                data.append(entry[i][0])
                targets.append(entry[i][1])

            tensors_data.append(torch.cat(data, dim=3))
            tensors_targets.append(torch.cat(targets, dim=0))

            if args.debug and idx > 10:
                break

        data = torch.cat(tensors_data, dim=0)
        targets = torch.stack(tensors_targets)
        dataset = torch.utils.data.TensorDataset(data, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return loader

    train_loader = make_multilabel_loader(train_loader, args.batch_size)
    test_loader = make_multilabel_loader(test_loader, 10000)
    return train_loader, test_loader


def _multi_mnist(path, max_digits=2, highest_digit=9, canvas_size=50, seed=42):
    """
    - Adapted from:  https://raw.githubusercontent.com/edwardlib/observations/master/observations/multi_mnist.py   
    - Allows overlapping digits and removes unnecessary big padding.

    Load the multiple MNIST data set [@eslami2016attend]. It modifies
  the original MNIST such that each image contains a number of
  non-overlapping random MNIST digits with equal probability.

  Args:
    path: str.
      Path to directory which either stores file or otherwise file will
      be downloaded and extracted there. Filename is
      `'multi_mnist_{}_{}_{}.npz'.format(max_digits, canvas_size, seed)`.
    max_digits: int, optional.
      Maximum number of non-overlapping MNIST digits per image to
      generate if not cached.
    highest_digit: int, optional.
      Highest digit: Sample from [0, 1, .., highest_digit].
    canvas_size: list of two int, optional.
      Width x height pixel size of generated images if not cached.
    seed: int, optional.
      Random seed to generate the data set from MNIST if not cached.

  Returns:
    Tuple of (np.ndarray of dtype uint8, list)
    `(x_train, y_train), (x_test, y_test)`. Each element in the y's is a
    np.ndarray of labels, one label for each digit in the image.
  """
    from scipy.misc import imresize

    def _sample_one(canvas_size, x_data, y_data):
        i = np.random.randint(x_data.shape[0])
        digit = x_data[i]
        label = y_data[i]
        scale = 0.1 * np.random.randn() + 1.3
        resized = imresize(digit, 1.0 / scale)
        width = resized.shape[0]
        padding = canvas_size - width
        pad_l = np.random.randint(0, padding)
        pad_r = np.random.randint(0, padding)
        pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
        positioned = np.pad(resized, pad_width, "constant", constant_values=0)
        return positioned, label

    def _sample_multi(num_digits, canvas_size, x_data, y_data):
        canvas_size_ext = canvas_size + 10
        canvas = np.zeros((canvas_size_ext, canvas_size_ext))
        labels = []
        for _ in range(num_digits):
            while True:
                positioned_digit, label = _sample_one(canvas_size_ext, x_data, y_data)

                if label not in labels:
                    canvas += positioned_digit
                    labels.append(label)
                    break

        if np.max(canvas) > 255 * 2:  # crude check for more than two overlapping digits
            return _sample_multi(num_digits, canvas_size, x_data, y_data)
        else:
            labels = np.array(labels, dtype=np.uint8)
            canvas = canvas[5:-5, 5:-5]
            return canvas, labels

    def _build_dataset(x_data, y_data, max_digits, canvas_size):
        x = []
        y = []
        data_size = x_data.shape[0]

        # Mask
        y_mask = y_data < highest_digit
        y_data = y_data[y_mask]
        x_data = x_data[y_mask]

        # Randomly sample k digits
        data_num_digits = np.random.randint(1, max_digits + 1, size=data_size)
        x_data = np.reshape(x_data, [x_data.shape[0], 28, 28])
        for num_digits in tqdm(data_num_digits):
            canvas, labels = _sample_multi(num_digits, canvas_size, x_data, y_data)
            x.append(canvas)
            y.append(labels)
        x = np.array(x, dtype=np.uint8)
        return x, y

    path = os.path.expanduser(path)
    cache_filename = "multi_mnist_max_digits={}_highest_digit={}_canvas={}_{}.npz".format(
        max_digits, highest_digit, canvas_size, seed
    )
    if os.path.exists(os.path.join(path, cache_filename)):
        logger.info("Loading cached Multi MNIST dataset: %s ...", cache_filename)
        data = np.load(os.path.join(path, cache_filename))
        return (data["x_train"], data["y_train"]), (data["x_test"], data["y_test"])
    logger.info("Creating new Multi MNIST dataset ...")
    np.random.seed(seed)
    (x_train, y_train), (x_test, y_test) = mnist(path)
    x_train, y_train = _build_dataset(x_train, y_train, max_digits, canvas_size)
    x_test, y_test = _build_dataset(x_test, y_test, max_digits, canvas_size)
    with open(os.path.join(path, cache_filename), "wb") as f:
        np.savez_compressed(
            f, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
        )
    return (x_train, y_train), (x_test, y_test)


def load_multi_mnist(n_labels, canvas_size=64, seed=0, args=None):
    """
    Load the multi mnist dataset, drawn on the given canvas size.

    Args:
        n_labels: Number of different possible digits (sample from [0, 1, ..., n_labels]).
        canvas_size: Canvas size.
        seed: Random seed.
        args: Command line arguments.
    Returns:
        Tuple: Train data loader, Test data loader
    """
    # Load the mnist dataset as np arrays
    (x_train, y_train), (x_test, y_test) = _multi_mnist(
        "./data/multi-mnist",
        max_digits=args.n_digits,
        highest_digit=n_labels,
        canvas_size=canvas_size,
        seed=seed,
    )

    if args.debug:
        x_train = x_train[:100]
        x_test = x_test[:100]
        y_train = y_train[:100]
        y_test = y_test[:100]

    if args.force_overfit:
        x_train = x_train[:100]
        y_train = y_train[:100]
        x_test = x_test[:100]
        y_test = y_test[:100]

    def _make_dataloader(x, y, batch_size):
        x = x / x.max()  # Squash between 0 and 1
        x = x - x_train.mean()  # Center
        x = x / x_train.std()  # Normalize
        x_torch = torch.tensor(x).float().unsqueeze(1)

        ys = []
        # Transform target labels into bincounts
        for y_i in y:
            y_count = torch.bincount(torch.tensor(y_i), minlength=n_labels)
            ys.append(y_count)

        y_torch = torch.stack(ys).float()
        dataset = torch.utils.data.TensorDataset(x_torch, y_torch)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return loader

    # batch_size = args.batch_size
    # test_batch_size = args.test_batch_size

    # If cuda is enabled, multiple batchsize by number of different devices
    # if args.cuda:
    #     num_cuda_devices = len(args.cuda_device_id)
    #     batch_size = batch_size * num_cuda_devices
    #     test_batch_size = test_batch_size * num_cuda_devices

    train_loader = _make_dataloader(x_train, y_train, args.batch_size)
    test_loader = _make_dataloader(x_test, y_test, args.test_batch_size)

    return train_loader, test_loader


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = _multi_mnist(
        "./data/multi-mnist", max_digits=5, highest_digit=10, canvas_size=50, seed=1
    )
    import matplotlib.pyplot as plt
    import os

    def save(x, y, dir):
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            plt.imshow(x_i)
            plt.axis("off")
            y_encoded = torch.bincount(torch.tensor(y_i), minlength=10)
            plt.title("Label: {}\nEncoded: {}".format(y_i, y_encoded))
            plt.savefig(os.path.join(dir, "%s.png" % i))

            if i > 10:
                break

    save(x_train, y_train, "data/multi-mnist/train")
    save(x_train, y_train, "data/multi-mnist/test")
