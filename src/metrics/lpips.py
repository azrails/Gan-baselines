import torch
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.metrics.base_metric import BaseMetric


class LpipsMetric(BaseMetric):
    def __init__(self, device):
        super().__init__(device)
        self.metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(
            self.device
        )

    def __call__(self, fake, real, *args, **kwargs):
        self.metric.update(self.denormalize(fake), real)

    def compute(self):
        metirc = self.metric.compute()
        self.metric.reset()
        return metirc

    def denormalize(self, x):
        return x.clamp(-1, 1)
