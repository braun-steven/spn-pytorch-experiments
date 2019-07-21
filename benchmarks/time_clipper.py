from src.models.pytorch import SPNClipper, SPNLayer
from src.models.models import SPNNeuron
from torch import nn
import torch
from torch.nn import functional as F
from time import time


class SPNNet(nn.Module):
    def __init__(self, in_features):
        super(SPNNet, self).__init__()

        # Define Layers
        self.fc1 = nn.Linear(in_features, 32)
        self.spn1 = SPNLayer(neuron=SPNNeuron, in_features=32, out_features=20)
        self.fc2 = nn.Linear(20, 10)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Reshape height and width into single dimension
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))  # First linear layer
        x = self.spn1(x)  # SPN
        x = self.fc2(x)  # Linear
        return F.log_softmax(x, dim=1)


def run(with_clip):
    spn = SPNNet(in_features=10)
    optimizer = torch.optim.Adam(spn.parameters())
    spn.train()
    clipper = SPNClipper("cpu")
    x = torch.randn(100, 10)
    for epoch in range(100):
        print("Epoch", epoch)
        optimizer.zero_grad()
        output = spn(x)
        loss = F.nll_loss(output, torch.randint(1, (100,)))
        loss.backward()
        optimizer.step()
        if with_clip:
            spn.apply(clipper)


t0 = time()
run(True)
t1 = time()
withclipper = t1 - t0


t0 = time()
run(False)
t1 = time()
withoutclipper = t1 - t0

print("With clipper:", withclipper)
print("Without clipper:", withoutclipper)
