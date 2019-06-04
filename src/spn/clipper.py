import torch
import torch.nn.functional as F

LOWER_BOUND = torch.tensor(1e-5)


class DistributionClipper(object):
    """
    Clip distribution parameters.
    """

    def __init__(self, device, lower_bound=1e-5):
        """
        Args:
            device: Torch device.
        """
        self.lower_bound = torch.tensor(lower_bound).to(device)

    def __call__(self, module):
        if hasattr(module, "stds"):
            param = module.stds.data
            param.clamp_(self.lower_bound)

        if hasattr(module, "concentration0"):
            param = module.concentration0.data
            param.clamp_(self.lower_bound)

        if hasattr(module, "concentration1"):
            param = module.concentration1.data
            param.clamp_(self.lower_bound)

        if hasattr(module, "concentration"):
            param = module.concentration.data
            param.clamp_(self.lower_bound)

        if hasattr(module, "rate"):
            param = module.rate.data
            param.clamp_(self.lower_bound)

        if hasattr(module, "df"):
            param = module.df.data
            param.clamp_(self.lower_bound)


class SumWeightClipper(object):
    """
    Clip sum weights > 0.
    """

    def __init__(self, device, lower_bound=1e-5):
        """
        Args:
            device: Torch device.
        """
        self.lower_bound = torch.tensor(lower_bound).to(device)

    def __call__(self, module):
        if hasattr(module, "sum_weights"):
            sum_weights = module.sum_weights.data
            sum_weights.clamp_(self.lower_bound)


class SumWeightNormalizer(object):
    """
    Normalize sum layer weights.
    """

    def __call__(self, module):
        if hasattr(module, "sum_weights"):
            sum_weights = module.sum_weights.data
            F.normalize(sum_weights, p=1, dim=2, out=sum_weights)
