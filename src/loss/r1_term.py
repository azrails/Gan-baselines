import torch
from torch import nn


class R1Regulaizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_value = None

    def forward(self, images, probs):
        grad = torch.autograd.grad(
            outputs=probs.sum(), inputs=images, create_graph=True
        )[0].pow(2)
        loss = 0.5 * grad.reshape(images.size(0), -1).sum(1).mean(0)
        self.loss_value = loss
        return loss
