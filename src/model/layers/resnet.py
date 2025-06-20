import torch
from torch import nn

from src.model.layers.initialization_utils import init_conv2d, init_fc_zero
from src.model.layers.residual import ResidualBlock


class ResNet(nn.Module):
    def __init__(
        self,
        scale_factor: int = 2,
        downsampling_block_size_expand: int = 4,
        downsampling_block_size_no_expand: int = 3,
        initial_hidden_channels: int = 64,
        out_dim: int = 64,
        domains: int = 2,
        compression_kernel: int = 4,
    ):
        super().__init__()
        max_channels = initial_hidden_channels * (
            scale_factor**downsampling_block_size_expand
        )
        self.in_block = nn.Conv2d(3, initial_hidden_channels, 1)
        self.activation = nn.LeakyReLU(0.2)
        self.downsample_block = nn.Sequential(
            *[
                ResidualBlock(
                    initial_hidden_channels * (scale_factor**i),
                    initial_hidden_channels * (scale_factor ** (i + 1)),
                    "downsample",
                )
                for i in range(downsampling_block_size_expand)
            ],
            *[
                ResidualBlock(max_channels, max_channels, "downsample")
                for i in range(downsampling_block_size_no_expand)
            ]
        )
        self.compression = nn.Conv2d(max_channels, max_channels, compression_kernel)
        self.projection = nn.ModuleList()
        for _ in range(domains):
            self.projection.append(nn.Linear(max_channels, out_dim))

        self.apply(init_conv2d)
        self.apply(init_fc_zero)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.in_block(x)
        out = self.activation(out)
        out = self.downsample_block(out)
        out = self.activation(out)
        out = self.compression(out)
        out = self.activation(out)
        out = out.view(y.size(0), -1)

        domain_outputs = []
        for layer in self.projection:
            domain_outputs.append(layer(out))

        idx = torch.arange(y.size(0), device=y.device)
        domain_outputs = torch.stack(domain_outputs, dim=1)
        out = domain_outputs[idx, y]
        return out
