import torch.nn.functional as F
from torch import nn


class RelativisticAdversarialLoss(nn.Module):
    """
    Wasserstein gan base adversarial loss.
    For more details https://en.wikipedia.org/wiki/Wasserstein_GAN
    """

    def __init__(self):
        super().__init__()
        self.loss_value = None

    def forward(self, logits):
        loss = F.softplus(-logits).mean()
        self.loss_value = loss
        return loss
