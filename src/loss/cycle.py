import torch
from torch import nn


class CycleLoss(nn.Module):
    """
    Cycle part of stargan loss.
    Measures distanse between initial image x and it cycled transformation.
    Forses bijection style-domain transformation.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
        self.loss_value = None

    def forward(
        self, sample: torch.Tensor, cycled_sample: torch.Tensor
    ) -> dict[str : torch.Tensor]:
        """
        Args:
            sample (torch.Tensor): sample of original images
            cycled_sample (torch.Tensor): sample of cycled images G(G(x, s'), s'')
                where s'' = E_y(x), y - original domain of x.

        Returns:
            (dict[str:torch.Tensor]): l1 cycle loss
        """
        loss = self.loss(sample, cycled_sample)
        self.loss_value = loss
        return loss
