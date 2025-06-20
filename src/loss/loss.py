import torch
from torch import nn

from src.loss.r1_term import R1Regulaizer


def param_scheduler(n_epoch, start, extra_coef=1750):
    param = torch.arange(start, 0, -1 / (n_epoch * extra_coef))
    for p in param:
        yield p
    while True:
        yield 0


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        adversarial,
        style_reconstruction,
        style_diversity,
        cycle,
        style_reconstruction_coef,
        cycle_coef,
        style_diversity_coef,
        n_epoch,
    ):
        super().__init__()
        self.adversarial = adversarial
        self.style_reconstruction = style_reconstruction
        self.style_diversity = style_diversity
        self.cycle = cycle
        self.style_reconstruction_coef = style_reconstruction_coef
        self.cycle_coef = cycle_coef
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
        *args,
        **kwargs
    ):
        loss = (
            self.adversarial(fake_logits)
            + self.style_reconstruction_coef
            * self.style_reconstruction(style_code, style_code_reconstruction)
            + self.cycle_coef * self.cycle(sample, sample_reconstruction)
            - next(self.style_diversity_coef)
            * self.style_diversity(styled_batch1, styled_batch2)
        )
        self.loss_value = loss
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, adversarial, r1_term, reg_coef=0):
        super().__init__()
        self.adversarial = adversarial
        self.loss_value = None
        self.r1_term = None
        self.reg_coef = reg_coef
        if r1_term:
            self.r1_term = R1Regulaizer()

    def forward(self, fake_logits, real_logits, real_images, *args, **kwargs):
        loss = -self.adversarial(fake_logits, real_logits)
        self.loss_value = loss
        if self.r1_term is not None:
            loss = loss + self.reg_coef * self.r1_term(real_images, real_logits)
        return loss
