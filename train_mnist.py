import torch
import torchvision
from torchvision import datasets, transforms
from observations import mnist
import numpy as np
from src.data.data_loader import get_mnist_loaders
from src.utils.utils import time_delta_now
from src.utils.utils import count_params
from src.spn.clipper import DistributionClipper
from src.spn.clipper import SumWeightNormalizer
from src.spn.clipper import SumWeightClipper
from src.spn.distributions import Normal
from src.spn.distributions import IsotropicMultivariateNormal
from src.spn.distributions import MultivariateNormal
from src.models.models import SPNNetPure
from src.models.models import SPNNeuronBig
from torch import nn
from torch.nn import functional as F
import time
import sys

import spn.algorithms.Inference as inference
import spn.io.Graphics as graphics


def one_hot(vector):
    result = np.zeros((vector.size, vector.max() + 1))
    result[np.arange(vector.size), vector] = 1
    return result


def get_mnist_loaders(use_cuda, device, batch_size):
    """
    Get the MNIST pytorch data loader.
    
    Args:
        use_cuda: Use cuda flag.

    """
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, download=True, transform=transformer),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, transform=transformer),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


def run_torch(n_epochs=100, batch_size=256):
    """Run the torch code.

    Args:
        n_epochs (int, optional): Number of epochs.
        batch_size (int, optional): Batch size.
    """
    from src.spn.rat_spn import RatSpnConstructor
    from torch import optim
    from torch import nn

    assert len(sys.argv) == 2, "Usage: train.mnist cuda/cpu"
    dev = sys.argv[1]

    rg = RatSpnConstructor(in_features=28 * 28, C=10, S=10, I=20, dropout=0.0)
    n_splits = 2
    for _ in range(0, n_splits):
        rg.random_split(2, 1)

    if dev == "cpu":
        device = torch.device("cpu")
        use_cuda = False
    else:
        device = torch.device("cuda:0")
        use_cuda = True
        torch.cuda.benchmark = True

    print("Using device:", device)

    model = rg.build().to(device)
    model.train()
    print(model)
    print(f"Layer 0: {count_params(model.region_spns[0]._leaf) * n_splits}")
    for i in range(1, len(model.region_spns[0]._inner_layers) + 1):
        print(
            f"Layer {i}: {count_params(model.region_spns[0]._inner_layers[i - 1]) * n_splits}"
        )
    print("Number of pytorch parameters: ", count_params(model))

    # Define optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    train_loader, test_loader = get_mnist_loaders(
        use_cuda, batch_size=batch_size, device=device
    )

    dist_clipper = DistributionClipper(device)
    log_interval = 100

    for epoch in range(n_epochs):
        t_start = time.time()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Send data to correct device
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)

            # Reset gradients
            optimizer.zero_grad()

            # Inference
            output = model(data)

            # Compute loss
            loss = loss_fn(output, target)

            # Backprop
            loss.backward()
            optimizer.step()

            # Clip distribution values and weights
            model.apply(dist_clipper)

            # Log stuff
            running_loss += loss.item()
            if batch_idx % log_interval == (log_interval - 1):
                pred = (
                    output.argmax(1).eq(target).sum().cpu().numpy()
                    / data.shape[0]
                    * 100
                )
                print(
                    "Train Epoch: {} [{: >5}/{: <5} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.0f}%".format(
                        epoch,
                        batch_idx * len(data),
                        60000,
                        100.0 * batch_idx / len(train_loader),
                        running_loss / log_interval,
                        pred,
                    ),
                    end="\r",
                )
                running_loss = 0.0

        t_delta = time_delta_now(t_start)
        print("Train Epoch: {} took {}".format(epoch, t_delta))
        if epoch % 5 == 4:
            print("Evaluating model ...")
            evaluate_model(model, device, train_loader, "Train")
            evaluate_model(model, device, test_loader, "Test")


def evaluate_model(model: torch.nn.Module, device, loader, tag) -> float:
    """
    Description for method evaluate_model.

    Args:
        model (nn.Module): PyTorch module.
        device: Execution device.
        loader: Data loader.
        tag (str): Tag for information.

    Returns:
        float: Tuple of loss and accuracy.
    """
    model.eval()
    loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

    loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)

    print(
        "{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            tag, loss, correct, len(loader.dataset), accuracy
        )
    )
    return (loss, accuracy)


if __name__ == "__main__":
    torch.cuda.benchmark = True
    run_torch(100, 100)
