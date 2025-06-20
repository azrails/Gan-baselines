import torch
from torch import nn


class EMA:
    def __init__(self, model, initial_decay=0.5, final_decay=0.999, num_epoch=100):
        self.model = model
        self.decay = initial_decay
        self.initial_decay = initial_decay
        self.ema_weights = {}
        self.backup = {}
        self.decay_rate = (final_decay - initial_decay) / num_epoch
        self.final_decay = final_decay
        self.epoch = 1

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_weights[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_weights[name] = torch.lerp(
                    param.data, self.ema_weights[name], self.decay
                )

    def from_pretrained(self, start_epoch, ema_weights=None):
        self.epoch = start_epoch
        if ema_weights is not None:
            self.ema_weights = ema_weights
        self.update_decay()

    def update_decay(self):
        self.epoch += 1
        self.decay = min(
            self.initial_decay + self.decay_rate * self.epoch, self.final_decay
        )

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.ema_weights[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup.clear()
