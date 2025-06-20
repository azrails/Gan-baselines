import torch
from torch import nn


class AdversarialLoss(nn.Module):
    """
    Wasserstein gan base adversarial loss.
    For more details https://en.wikipedia.org/wiki/Wasserstein_GAN
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.loss_value = None

    def forward(
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None
    ):

        targets = torch.ones_like(fake_logits)
        if real_logits is None:
            loss = self.loss(fake_logits, targets)
            self.loss_value = loss
        else:
            fake_targets = torch.zeros_like(fake_logits)
            loss = -(
                self.loss(real_logits, targets) + self.loss(fake_logits, fake_targets)
            )
            self.loss_value = -loss
        return loss
