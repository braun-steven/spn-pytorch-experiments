import unittest

import numpy as np

import torch
from torch import optim
from spn.structure.Base import Product, Sum
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
from spn.gpu.TensorFlow import spn_to_tf_graph, optimize_tf_graph
from spn.gpu.TensorFlow import eval_tf
from spn.algorithms.Inference import log_likelihood
from torch import nn

from pytorch import CategoricalNode, GaussianNode, ProductNode, SumNode, optimize_torch

DELTA = 1e-10


class IdentityLeaf(nn.Module):
    """Identity leaf node (in log space) for testing purposes."""

    def __init__(self, scope):
        super(IdentityLeaf, self).__init__()
        self.scope = scope

    def forward(self, x):
        return torch.log(x[:, self.scope])


class TestPytorchNodes(unittest.TestCase):
    """Test cases for different node types."""

    def test_sum_node(self):
        """Test SumNode implementation."""
        # Init product node
        id0 = IdentityLeaf(scope=0)
        id1 = IdentityLeaf(scope=1)
        id2 = IdentityLeaf(scope=2)
        children = [id0, id1, id2]
        weights = np.array([0.1, 0.4, 0.5])
        sumnode = SumNode(children=children, weights=weights)

        # Define input: Two samples with three features
        sample1 = np.array([1, 2, 3])
        sample2 = np.array([10, 20, 30])
        x = torch.Tensor([sample1, sample2])

        # Get sum node result
        result = sumnode(x)

        # Expected results
        expected_result = np.log([np.sum(weights * sample1), np.sum(weights * sample2)]).tolist()

        # Assertions
        self.assertEqual(len(result.tolist()), 2)
        self.assertTrue(np.isclose(result.tolist(), expected_result, atol=DELTA).all())

    def test_product_node(self):
        """Test product node implementation"""
        # Init product node
        id0 = IdentityLeaf(scope=0)
        id1 = IdentityLeaf(scope=1)
        id2 = IdentityLeaf(scope=2)
        prod = ProductNode(children=[id0, id1, id2])

        # Define input: Two samples with three features
        sample1 = np.array([1, 2, 3])
        sample2 = np.array([10, 20, 30])
        x = torch.Tensor([sample1, sample2])

        # Get product node result
        result = prod(x)

        # Product in logspace is sum
        expected_result = [np.sum(np.log(sample1)), np.sum(np.log(sample2))]

        # Assertions
        self.assertEqual(len(result.tolist()), 2)
        self.assertTrue(np.isclose(result.tolist(), expected_result, atol=DELTA).all())

    def test_gaussian_node(self):
        """Test the GaussianNode implementation"""
        means = [0.0, 0.5, 1.0]
        stds = [1.0, 2.0, 3.0]
        gauss0 = GaussianNode(mean=means[0], std=stds[0], scope=0)
        gauss1 = GaussianNode(mean=means[1], std=stds[1], scope=1)
        gauss2 = GaussianNode(mean=means[2], std=stds[2], scope=2)
        sample1 = np.array([1, 2, 3])
        sample2 = np.array([10, 20, 30])
        x = torch.Tensor([sample1, sample2])

        # Get results
        res_gauss0 = gauss0(x)
        res_gauss1 = gauss1(x)
        res_gauss2 = gauss2(x)

        # Expect results from normal distributions
        normal0 = torch.distributions.Normal(loc=means[0], scale=stds[0])
        normal1 = torch.distributions.Normal(loc=means[1], scale=stds[1])
        normal2 = torch.distributions.Normal(loc=means[2], scale=stds[2])

        exp_gauss0 = normal0.log_prob(torch.Tensor([1, 10]))
        exp_gauss1 = normal1.log_prob(torch.Tensor([2, 20]))
        exp_gauss2 = normal2.log_prob(torch.Tensor([3, 30]))

        # Assertions
        self.assertEqual(len(res_gauss0.tolist()), 2)
        self.assertEqual(len(res_gauss1.tolist()), 2)
        self.assertEqual(len(res_gauss2.tolist()), 2)

        # Assert that results are numerically equal
        self.assertTrue(np.isclose(res_gauss0.tolist(), exp_gauss0, atol=DELTA).all())
        self.assertTrue(np.isclose(res_gauss1.tolist(), exp_gauss1, atol=DELTA).all())
        self.assertTrue(np.isclose(res_gauss2.tolist(), exp_gauss2, atol=DELTA).all())

    def test_equal_to_tf(self):
        # SPFLow implementation
        g00 = Gaussian(mean=0.0, stdev=1.0, scope=0)
        g10 = Gaussian(mean=1.0, stdev=2.0, scope=1)
        g01 = Gaussian(mean=3.0, stdev=2.0, scope=0)
        g11 = Gaussian(mean=5.0, stdev=1.0, scope=1)
        p0 = Product(children=[g00, g10])
        p1 = Product(children=[g01, g11])
        s = Sum(weights=[0.2, 0.8], children=[p0, p1])

        assign_ids(s)
        rebuild_scopes_bottom_up(s)

        # Test for 100 random samples
        data = np.random.randn(100, 2)

        # LL from SPN
        ll = log_likelihood(s, data)

        # PyTorch implementation
        g00 = GaussianNode(mean=0.0, std=1.0, scope=0)
        g10 = GaussianNode(mean=1.0, std=2.0, scope=1)
        g01 = GaussianNode(mean=3.0, std=2.0, scope=0)
        g11 = GaussianNode(mean=5.0, std=1.0, scope=1)
        p0 = ProductNode(children=[g00, g10])
        p1 = ProductNode(children=[g01, g11])
        rootnode = SumNode(weights=[0.2, 0.8], children=[p0, p1])

        datatensor = torch.Tensor(data)
        # LL from pytorch
        ll_torch = rootnode(datatensor)

        # Assert equality
        self.assertTrue(np.isclose(np.array(ll).squeeze(), ll_torch.detach().numpy(), atol=DELTA).all())

    def test_spn_to_torch(self):
        # SPFLow implementation
        n0 = Gaussian(mean=0.0, stdev=1.0, scope=0)
        n1 = Categorical(p=[0.1, 0.3, 0.6])
        n2 = Sum(weights=[0.1, 0.2, 0.3, 0.4], children=[n0, n1])
        n3 = Product(children=[n0, n1])

        torch_n0 = GaussianNode.from_spn(n0)
        torch_n1 = CategoricalNode.from_spn(n1)
        torch_n2 = SumNode.from_spn(n2)
        torch_n3 = ProductNode.from_spn(n3)

        self.assertEqual(torch_n0.mean, n0.mean)
        self.assertEqual(torch_n0.std, n0.stdev)
        self.assertTrue(np.isclose(torch_n1.p.detach().numpy(), n1.p, atol=DELTA).all())
        self.assertTrue(np.isclose(torch_n2.weights.detach().numpy(), n2.weights, atol=DELTA).all())

    def test_torch_vs_tf_time(self):
        # Create sample data
        from sklearn.datasets.samples_generator import make_blobs
        import tensorflow as tf
        from time import time

        X, y = make_blobs(n_samples=10, centers=3, n_features=2, random_state=0)
        X = X.astype(np.float32)

        # SPFLow implementation
        g00 = Gaussian(mean=0.0, stdev=1.0, scope=0)
        g10 = Gaussian(mean=1.0, stdev=2.0, scope=1)
        g01 = Gaussian(mean=3.0, stdev=2.0, scope=0)
        g11 = Gaussian(mean=5.0, stdev=1.0, scope=1)
        p0 = Product(children=[g00, g10])
        p1 = Product(children=[g01, g11])
        s = Sum(weights=[0.2, 0.8], children=[p0, p1])
        assign_ids(s)
        rebuild_scopes_bottom_up(s)

        # Convert
        tf_spn, data_placeholder, variable_dict = spn_to_tf_graph(s, data=X)
        torch_spn = SumNode.from_spn(s)

        # Optimizer
        lr = 0.001
        tf_optim = tf.train.AdamOptimizer(lr)
        torch_optim = optim.Adam(torch_spn.parameters(), lr)

        t0 = time()
        epochs = 10
        optimize_tf_graph(tf_spn, variable_dict, data_placeholder, X, epochs=epochs, optimizer=tf_optim)
        t1 = time()
        optimize_torch(torch_spn, X, epochs=epochs, optimizer=torch_optim)
        t2 = time()

        print("Tensorflow took: ", t1 - t0)
        print("PyTorch took: ", t2 - t1)


if __name__ == "__main__":
    unittest.main()
