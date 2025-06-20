import torch
from torch import nn

from src.loss import R1Regulaizer
from src.loss.relativistic_adversarial import RelativisticAdversarialLoss


def param_scheduler(n_epoch, start, extra_coef=1750):
    param = torch.arange(start, 0, -1 / (n_epoch * extra_coef))
    for p in param:
        yield p
    while True:
        yield 0


class RelativisticGeneratorLoss(nn.Module):
    def __init__(
        self,
        style_reconstruction,
        style_diversity,
        cycle,
        style_reconstruction_coef,
        cycle_coef,
        style_diversity_coef,
        n_epoch,
    ):
        super().__init__()
        self.style_reconstruction = style_reconstruction
        self.style_diversity = style_diversity
        self.cycle = cycle
        self.style_reconstruction_coef = style_reconstruction_coef
        self.cycle_coef = cycle_coef
        self.adversarial = RelativisticAdversarialLoss()
        self.loss_value = None
        self.style_diversity_init_coef = style_diversity_coef
        self.n_epoch = n_epoch
        self.style_diversity_coef = param_scheduler(n_epoch, style_diversity_coef)

    def set_steps(self, epoch_len=1750):
        self.style_diversity_coef = param_scheduler(
            self.n_epoch, self.style_diversity_init_coef, epoch_len
        )

    def forward(
        self,
        fake_logits,
        style_code,
        style_code_reconstruction,
        styled_batch1,
        styled_batch2,
        sample,
        sample_reconstruction,
        real_logits,
    ):

        loss = (
            self.adversarial(fake_logits - real_logits)
            + self.style_reconstruction_coef
            * self.style_reconstruction(style_code, style_code_reconstruction)
            + self.cycle_coef * self.cycle(sample, sample_reconstruction)
            - next(self.style_diversity_coef)
            * self.style_diversity(styled_batch1, styled_batch2)
        )
        self.loss_value = loss
        return loss


class RelativisticDiscriminatorLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.adversarial = RelativisticAdversarialLoss()
        self.reg1 = R1Regulaizer()
        self.reg2 = R1Regulaizer()
        self.loss_value = None
        self.beta = beta

    def forward(self, fake_logits, real_logits, real_images, fake_images):
        loss = self.adversarial(real_logits - fake_logits) + (self.beta / 2) * (
            self.reg1(real_images, real_logits) + self.reg2(fake_images, fake_logits)
        )
        self.loss_value = loss
        return loss
