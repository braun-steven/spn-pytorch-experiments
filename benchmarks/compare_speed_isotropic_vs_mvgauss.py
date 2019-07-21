import sys
from matplotlib import pyplot as plt
import torch
from torch import nn
from src.spn.layers import Product
from src.spn.distributions import Normal, MultivariateNormal
from src.spn.distributions import Leaf
from src.utils.utils import time_delta_now
import time


class IndependentNormal(Leaf):
    def __init__(self, multiplicity, in_features, cardinality, dropout=0.0):
        """
        Create multivariate normal that only has non zero values in the covariance matrix on the diagonal.

        Args:
            multiplicity: Number of parallel representations for each input feature.
            cardinality: Number of variables per gauss.
            in_features: Number of input features.
            droptout: Dropout probabilities.
        """
        super(IndependentNormal, self).__init__(multiplicity, in_features, dropout)
        self.gauss = Normal(
            multiplicity=multiplicity, in_features=in_features, dropout=dropout
        )
        self.prod = Product(in_features=in_features, cardinality=cardinality)

        self.cardinality = cardinality

    def forward(self, x):
        x = self.gauss(x)
        x = self.prod(x)
        return x


def run(Model, multiplicity, in_features, cardinality, iterations):
    model = Model(multiplicity, in_features, cardinality).to(DEVICE)
    if Model == MultivariateNormal:
        model._mv.loc = model.means
    # Use SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Send data to correct device
    data = torch.randn(batch_size, in_features).to(DEVICE)

    # Train for 100 iterations
    tstart = time.time()
    for i in range(iterations):

        # Reset gradients
        optimizer.zero_grad()

        # Inference
        output = model(data)

        # Comput loss
        loss = -1 * output.mean()

        # Backprop
        loss.backward()
        optimizer.step()

    return time.time() - tstart


def compare_cov_size():
    # Compare different number of input features with fixed cov size 4
    t_ins = []
    t_mns = []
    # 2**12 x 2**12 cov matrix does not fit into the memory => 11 is the limit
    xs = [2 ** i for i in range(2, 12)]
    n_features = 2 ** 14
    for cardinality in xs:
        print("cardinality =", cardinality)
        # Run a few times before measuring
        run(IndependentNormal, multiplicity, n_features, cardinality, iterations=5)
        t_ins.append(
            run(IndependentNormal, multiplicity, n_features, cardinality, iterations)
        )

        # Run a few times before measuring
        run(MultivariateNormal, multiplicity, n_features, cardinality, iterations=5)
        t_mns.append(
            run(MultivariateNormal, multiplicity, n_features, cardinality, iterations)
        )

    plt.figure()
    plt.plot(xs, t_ins, label="IndependentNormal")
    plt.plot(xs, t_mns, label="MultivariateNormal")
    plt.xscale("log", basex=2)
    plt.xlabel("Number of Variables per Gaussian")
    plt.ylabel("Time in Seconds")
    plt.title(
        f"Independent Normal vs MV-Gauss: Gauss-Size vs Time\n($features = {n_features}$, $bs={batch_size}$, $iters={iterations}$)"
    )
    plt.legend()
    plt.savefig("./comp-independent-normal-vs-mv-gauss/gauss-size.png", dpi=180)


def compare_in_features():
    # Compare different number of input features with fixed cov size 4
    t_ins = []
    t_mns = []
    feats = [2 ** i for i in range(4, 15)]
    cardinality = 16
    for in_features in feats:
        print("in_features =", in_features)

        run(MultivariateNormal, multiplicity, in_features, cardinality, iterations=5)
        t_mns.append(
            run(MultivariateNormal, multiplicity, in_features, cardinality, iterations)
        )

        run(IndependentNormal, multiplicity, in_features, cardinality, iterations=5)
        t_ins.append(
            run(IndependentNormal, multiplicity, in_features, cardinality, iterations)
        )

    plt.figure()
    plt.plot(feats, t_ins, label="IndependentNormal")
    plt.plot(feats, t_mns, label="MultivariateNormal")
    plt.xscale("log", basex=2)
    plt.xlabel("Number of Features")
    plt.ylabel("Time in Seconds")
    plt.title(
        f"Independent Normal vs MV-Gauss: Features vs Time\n($covsize = {cardinality}$, $bs={batch_size}$, $iters={iterations}$)"
    )
    plt.legend()
    plt.savefig("./comp-independent-normal-vs-mv-gauss/nfeats.png", dpi=180)


if __name__ == "__main__":
    batch_size = 256
    multiplicity = 10
    iterations = 1000

    assert len(sys.argv) == 2, "Usage: train.mnist cuda/cpu"
    dev = sys.argv[1]

    if dev == "cpu":
        DEVICE = torch.device("cpu")
        use_cuda = False
    else:
        DEVICE = torch.device("cuda:0")
        torch.cuda.benchmark = True

    compare_in_features()
    compare_cov_size()
