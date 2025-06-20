import torch
from torch import nn
from torchmetrics.image.inception import InceptionScore

from src.metrics.base_metric import BaseMetric


class IsMetric(BaseMetric):
    def __init__(self, device):
        super().__init__(device)
        self.metric = InceptionScore().to(self.device)

    def __call__(self, fake, *args, **kwargs):
        self.metric.update(self.denormalize(fake))

    def compute(self):
        metirc = self.metric.compute()
        self.metric.reset()
        return metirc[0]

    def denormalize(self, imgs):
        imgs = (imgs + 1) / 2
        imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)
        return imgs
