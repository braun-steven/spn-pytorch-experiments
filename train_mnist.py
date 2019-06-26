import torch
import torchvision
from torchvision import datasets, transforms
from observations import mnist
import tensorflow as tf
import numpy as np
import spn.experiments.RandomSPNs.RAT_SPN as RAT_SPN
import spn.experiments.RandomSPNs.region_graph as region_graph
from src.data.data_loader import get_mnist_loaders
from src.utils.utils import time_delta_now
from src.utils.utils import count_params
from src.spn.clipper import DistributionClipper
from src.spn.clipper import SumWeightNormalizer
from src.spn.clipper import SumWeightClipper
from src.spn.distributions import Normal
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


def load_mnist():
    (train_im, train_lab), (test_im, test_lab) = mnist("data/mnist")
    train_im_mean = np.mean(train_im, 0)
    train_im_std = np.std(train_im, 0)
    std_eps = 1e-7
    train_im = (train_im - train_im_mean) / (train_im_std + std_eps)
    test_im = (test_im - train_im_mean) / (train_im_std + std_eps)

    # train_im /= 255.0
    # test_im /= 255.0
    return (train_im, train_lab), (test_im, test_lab)


def train_spn(
    spn, train_im, train_lab=None, num_epochs=50, batch_size=100, sess=tf.Session()
):

    input_ph = tf.placeholder(tf.float32, [batch_size, train_im.shape[1]])
    label_ph = tf.placeholder(tf.int32, [batch_size])
    marginalized = tf.zeros_like(input_ph)
    spn_output = spn.forward(input_ph, marginalized)
    if train_lab is not None:
        disc_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label_ph, logits=spn_output
            )
        )
        label_idx = tf.stack([tf.range(batch_size), label_ph], axis=1)
        gen_loss = tf.reduce_mean(-1 * tf.gather_nd(spn_output, label_idx))
    very_gen_loss = -1 * tf.reduce_mean(tf.reduce_logsumexp(spn_output, axis=1))
    loss = disc_loss
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    batches_per_epoch = train_im.shape[0] // batch_size

    # sess.run(tf.variables_initializer(optimizer.variables()))
    sess.run(tf.global_variables_initializer())

    for i in range(num_epochs):
        t_start = time.time()
        num_correct = 0
        for j in range(batches_per_epoch):
            im_batch = train_im[j * batch_size : (j + 1) * batch_size, :]
            label_batch = train_lab[j * batch_size : (j + 1) * batch_size]

            _, cur_output, cur_loss = sess.run(
                [train_op, spn_output, loss],
                feed_dict={input_ph: im_batch, label_ph: label_batch},
            )

            max_idx = np.argmax(cur_output, axis=1)

            num_correct_batch = np.sum(max_idx == label_batch)
            num_correct += num_correct_batch
        t_delta = time_delta_now(t_start)
        print("Train Epoch: {} took {}".format(i, t_delta))

        acc = num_correct / (batch_size * batches_per_epoch)
        print(i, acc, cur_loss)


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def run_tf():
    rg = region_graph.RegionGraph(range(28 * 28))
    for _ in range(0, 1):
        rg.random_split(2, 1)

    num_epochs = 100
    args = RAT_SPN.SpnArgs()
    args.normalized_sums = True
    args.num_sums = 2
    args.num_univ_distros = 20
    spn = RAT_SPN.RatSpn(10, region_graph=rg, name="obj-spn", args=args)
    print("num_params", spn.num_params())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    from spn.io.Graphics import plot_spn
    from matplotlib import pyplot as plt

    (train_im, train_labels), _ = load_mnist()
    train_spn(spn, train_im, train_labels, num_epochs=num_epochs, sess=sess)

    # dummy_input = np.random.normal(0.0, 1.2, [10, 9])
    dummy_input = train_im[:5]
    input_ph = tf.placeholder(tf.float32, dummy_input.shape)
    output_tensor = spn.forward(input_ph)
    tf_output = sess.run(output_tensor, feed_dict={input_ph: dummy_input})

    output_nodes = spn.get_simple_spn(sess)
    simple_output = []
    for node in output_nodes:
        simple_output.append(inference.log_likelihood(node, dummy_input)[:, 0])
    # graphics.plot_spn2(output_nodes[0])
    # graphics.plot_spn_to_svg(output_nodes[0])
    simple_output = np.stack(simple_output, axis=-1)
    print(tf_output, simple_output)
    simple_output = softmax(simple_output, axis=1)
    tf_output = softmax(tf_output, axis=1) + 1e-100
    print(tf_output, simple_output)
    relative_error = np.abs(simple_output / tf_output - 1)
    print(np.average(relative_error))


def get_mnist_loaders(use_cuda, batch_size):
    """
    Get the MNIST pytorch data loader.
    
    Args:
        use_cuda: Use cuda flag.

    """

    kwargs = {"num_workers": 1, "pin_memory": False} if use_cuda else {}

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


def run_torch(n_epochs=100, batch_size=100):
    from src.spn.rat_spn import RatSpnConstructor
    from torch import optim
    from torch import nn

    dev = "cpu"

    rg = RatSpnConstructor(in_features=28 * 28, C=10, S=2, I=20, dropout=0.0)
    for _ in range(0, 1):
        rg.random_split(2, 1)

    if dev == "cpu":
        device = torch.device("cpu")
        use_cuda = False
    else:
        device = torch.device("cuda:0")
        use_cuda = True

    model = rg.build().to(device)
    # TODO: model.spns[0][0].gauss.loc.device is CPU even when cuda is selected
    # for i in range(len(model.spns)):
    #     model.spns[i][0]._mv.loc = model.spns[i][0].means
    for i in range(len(model.spns)):
        leafs = model.spns[i][0].gauss
        # model.spns[i][0].gauss.loc = model.spns[i][0].means
        # leafs.cov_diag = model.spns[i][0].stds
        # leafs._unbroadcasted_cov_diag = model.spns[i][0].stds
        leafs._unbroadcasted_cov_diag = leafs._unbroadcasted_cov_diag.to(device)
        leafs.loc = leafs.loc.to(device)
        leafs.cov_diag = leafs.cov_diag.to(device)
        leafs._unbroadcasted_cov_factor = leafs._unbroadcasted_cov_factor.to(device)
        leafs._capacitance_tril = leafs._capacitance_tril.to(device)

    model.train()
    print(model)
    print("Number of pytorch parameters: ", count_params(model))

    # Define optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loader, test_loader = get_mnist_loaders(use_cuda, batch_size=batch_size)

    dist_clipper = DistributionClipper(device)

    for epoch in range(n_epochs):
        t_start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Send data to correct device
            data, target = data.to(device), target.to(device)
            data = data.view(train_loader.batch_size, -1)

            # print(model.spns[0][0].gauss.loc)
            # Reset gradients
            optimizer.zero_grad()

            # Inference
            output = model(data)

            # Compute loss
            # N = data.shape[0]
            # z = torch.logsumexp(output, dim=1)
            # loss = -1 / N * (output[np.arange(N), target] - z).sum()
            # Comput loss
            loss = loss_fn(output, target)

            # plot_sample(data[0].view(28, 28), target[0], output[0], loss)

            # Backprop
            loss.backward(retain_graph=True)
            optimizer.step()

            # Clip distribution values and weights
            model.apply(dist_clipper)

            # Log stuff
            if batch_idx % 10 == 0:
                print(
                    "Train Epoch: {} [{: >5}/{: <5} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        60000,
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
            # if batch_idx % log_interval * 3 == 0:
            #     logger.info("Samples:")
            #     logger.info("Target: %s", target[0].cpu().numpy())
            #     logger.info("Output: %s", output[0].detach().cpu().numpy())

        t_delta = time_delta_now(t_start)
        print("Train Epoch: {} took {}".format(epoch, t_delta))
        # evaluate_model(model, device, train_loader, "Train")


def plot_sample(x, y, y_pred, loss):
    """
    Plot a single sample witht the target and prediction in the title.

    Args:
        x: Image.
        y: Target.
        y_pred: Target prediction.
        loss: Loss value.
    """
    import matplotlib.pyplot as plt

    plt.imshow(x.squeeze().numpy())
    plt.title(
        "y={}\ny_pred={}\nloss={}".format(
            y.squeeze().numpy(),
            y_pred.squeeze().detach().numpy(),
            loss.detach().numpy(),
        )
    )
    plt.show()


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
    criterion = nn.NLLLoss()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.view(loader.batch_size, -1)
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


def run_torch_spn(n_epochs, batch_size):
    from torch import optim
    from torch import nn

    dev = sys.argv[1]

    if dev == "cpu":
        device = torch.device("cpu")
        use_cuda = False
    else:
        device = torch.device("cuda:0")
        use_cuda = True

    class CustomLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(28 * 28, 32)
            self.linear2 = nn.Linear(32, 10)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = self.linear(x)
            x = nn.functional.relu(x)
            x = self.linear2(x)
            x = nn.functional.relu(x)

            return x.sigmoid()

    class CustomSPN(nn.Module):
        def __init__(self):
            from src.spn import layers

            super().__init__()
            ch = [2 ** i for i in range(4, 0, -1)]
            ch = [1, 1, 1, 1]
            card = 2
            n_gaussians = 5
            in_features = 28 * 28
            self.l1 = Normal(multiplicity=n_gaussians, in_features=in_features)
            self.sum1 = layers.Sum(n_gaussians, in_features, 1)
            self.prod1 = layers.Product(in_features, 2)
            self.sum2 = layers.Sum(1, 28 * 14, 1)
            self.prod2 = layers.Product(28 * 14, 2)
            self.sum3 = layers.Sum(1, 28 * 7, 1)
            self.prod3 = layers.Product(28 * 7, 2)
            self.sum4 = layers.Sum(1, 14 * 7, 1)
            self.prod4 = layers.Product(14 * 7, 2)
            self.sum5 = layers.Sum(1, 7 * 7, 1)
            self.prod5 = layers.Product(7 * 7, 7 * 7)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = self.l1(x)
            x = self.sum1(x)
            x = self.prod1(x)
            x = self.sum2(x)
            x = self.prod2(x)
            x = self.sum3(x)
            x = self.prod3(x)
            x = self.sum4(x)
            x = self.prod4(x)
            x = self.sum5(x)
            x = self.prod5(x)
            x = x.squeeze(1)
            return x

    class CustomSPNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.spns = nn.ModuleList([CustomSPN() for i in range(10)])

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            res = torch.stack([spn(x) for spn in self.spns], dim=1)
            # spn_results = [spn(x) for spn in self.spns]
            # x = torch.stack(spn_results, dim=1)
            N = x.shape[0]
            z = torch.logsumexp(res, dim=1)
            y = res.squeeze(2) - z
            return y

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = x.view(x.shape[0], 1, 28, 28)
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

            return F.log_softmax(x, dim=1)

    # model = SPNNetPure(28 * 28, 10, spnneuron=SPNNeuronBig).to(device)
    model = CustomSPNNet().to(device)
    # model = CustomLinear().to(device)
    # model = Net().to(device)

    model.train()
    print(model)
    print("Number of pytorch parameters: ", count_params(model))

    # Define optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_loader, test_loader = get_mnist_loaders(use_cuda, batch_size=batch_size)

    dist_clipper = DistributionClipper(device)

    for epoch in range(n_epochs):
        t_start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Send data to correct device
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)

            # print(model.spns[0][0].gauss.loc)
            # Reset gradients
            t0 = time.time()
            optimizer.zero_grad()
            # print("optimzer zero_grad", time_delta_now(t0))

            # Inference
            t0 = time.time()
            output = model(data)
            # print("Forward pass", time_delta_now(t0))

            # Compute loss
            # N = data.shape[0]
            # z = torch.logsumexp(output, dim=1)
            # loss = -1 / N * (output[np.arange(N), target] - z).sum()
            # Comput loss
            loss = loss_fn(output, target)
            # t0 = time.time()
            # N = data.shape[0]
            # z = torch.logsumexp(output, dim=1)
            # loss = -1 / N * (output[np.arange(N), target] - z).sum()
            # print("Loss calc", time_delta_now(t0))

            # plot_sample(data[0].view(28, 28), target[0], output[0], loss)

            # Backprop
            t0 = time.time()
            loss.backward()
            optimizer.step()
            # print("Backward pass", time_delta_now(t0))

            # Clip distribution values and weights
            t0 = time.time()
            model.apply(dist_clipper)
            # print("Clipper", time_delta_now(t0))

            # Log stuff
            if batch_idx % 10 == 0:
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
                        loss.item(),
                        pred,
                    )
                )
            # if batch_idx % log_interval * 3 == 0:
            #     logger.info("Samples:")
            #     logger.info("Target: %s", target[0].cpu().numpy())
            #     logger.info("Output: %s", output[0].detach().cpu().numpy())

        t_delta = time_delta_now(t_start)
        print("Train Epoch: {} took {}".format(epoch, t_delta))
        # evaluate_model(model, device, train_loader, "Train")


if __name__ == "__main__":
    # run_tf()
    # torch.autograd.set_detect_anomaly(True)
    # run_torch(100, 100)
    run_torch_spn(100, 512)
