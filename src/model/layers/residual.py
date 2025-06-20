import torch
from torch import nn

from src.model.layers.adaIn import AdaIn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_type: str | None = None,
        norm_type: str = "in",
        embedding_dim: int | None = None,
    ):
        super().__init__()
        self.block_type = block_type
        self.norm_type = norm_type == "in"

        # chose block type
        if block_type == "upsample":
            self.interpolation = nn.Upsample(scale_factor=2)
        elif block_type == "downsample":
            self.interpolation = nn.AvgPool2d(2)

        # chose norm type
        if self.norm_type:
            self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(in_channels, affine=True)
        else:
            assert (
                embedding_dim is not None
            ), "Embeding dim must be initialized explicity"
            self.norm1 = AdaIn(in_channels, embedding_dim)
            self.norm2 = AdaIn(in_channels, embedding_dim)

        # another parameters
        self.activation = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.train_addition = out_channels != in_channels
        if self.train_addition:
            self.addition = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
        identity = x

        if self.norm_type:
            out = self.norm1(x)
        else:
            out = self.norm1(x, s)
        out = self.activation(out)
        # upsample / downsample
        if self.block_type is not None:
            out = self.interpolation(out)
        out = self.conv1(out)

        if self.norm_type:
            out = self.norm2(out)
        else:
            out = self.norm2(out, s)
        out = self.activation(out)
        out = self.conv2(out)

        # adding extra chanels to x and upsample / downsample
        if self.block_type is not None:
            identity = self.interpolation(identity)
        if self.train_addition:
            identity = self.addition(identity)

        # normalizing variance
        return (out + identity) / (2 ** (1 / 2))
