import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance

from src.metrics.base_metric import BaseMetric


class FidMetric(BaseMetric):
    def __init__(self, device):
        super().__init__(device)
        self.metric = (
            FrechetInceptionDistance().set_dtype(torch.float32).to(self.device)
        )

    def __call__(self, fake, real, *args, **kwargs):
        self.metric.update(self.denormalize(fake), real=False)
        self.metric.update(self.denormalize(real), real=True)

    def compute(self):
        metirc = self.metric.compute()
        self.metric.reset()
        return metirc

    def denormalize(self, imgs):
        imgs = (imgs + 1) / 2
        imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)
        return imgs
