import torch
from torch import nn

from src.model.layers.initialization_utils import init_conv2d
from src.model.layers.residual import ResidualBlock
from src.model.layers.transformer import Transformer


class Generator(nn.Module):
    def __init__(
        self,
        scale_factor: int = 2,
        downsampling_block_size_expand: int = 4,
        downsampling_block_size_no_expand: int = 1,
        middle_block_size: int = 4,
        initial_hidden_channels: int = 64,
        embedding_dim: int = 128,
    ):
        super().__init__()
        assert (
            middle_block_size % 2 == 0
        ), "Warning middle block size must be devided by 2"
        middle_channels = initial_hidden_channels * (
            scale_factor**downsampling_block_size_expand
        )
        inverse_scale_factor = 1 / scale_factor

        self.in_block = nn.Conv2d(3, initial_hidden_channels, 3, 1, 1)
        self.out_block = nn.Sequential(
            nn.InstanceNorm2d(initial_hidden_channels, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(initial_hidden_channels, 3, 1),
        )
        self.downsample_block = self._make_block(
            initial_hidden_channels,
            scale_factor,
            downsampling_block_size_expand,
            "downsample",
        )
        self.downsample_block.extend(
            self._make_block(
                middle_channels,
                1,
                downsampling_block_size_no_expand,
                "downsample",
            )
        )
        self.downsample_block.extend(
            nn.ModuleList([Transformer(int(middle_channels), 8, 64, 4)])
        )

        self.upsample_block = nn.ModuleList([Transformer(int(middle_channels), 8, 64, 4)])

        self.upsample_block.extend(
            self._make_block(
            middle_channels,
            1,
            downsampling_block_size_no_expand,
            "upsample",
            norm_type="adain",
            embeding_dim=embedding_dim,
        ))
        self.upsample_block.extend(
            self._make_block(
                middle_channels,
                inverse_scale_factor,
                downsampling_block_size_expand,
                "upsample",
                norm_type="adain",
                embeding_dim=embedding_dim,
            )
        )

        self.middle_block1 = self._make_block(
            middle_channels, 1, middle_block_size // 2
        )
        self.middle_block2 = self._make_block(
            middle_channels,
            1,
            middle_block_size // 2,
            norm_type="adain",
            embeding_dim=embedding_dim,
        )
        self.apply(init_conv2d)

    def _make_block(
        self,
        in_channels: int,
        scale_factor: int = 1,
        block_size: int = 4,
        block_type: str | None = None,
        norm_type: str = "in",
        embeding_dim: int | None = None,
    ):
        layers = []
        for i in range(block_size):
            layers.append(
                ResidualBlock(
                    int(in_channels * (scale_factor**i)),
                    int(in_channels * (scale_factor ** (i + 1))),
                    block_type,
                    norm_type,
                    embeding_dim,
                )
            )
            if block_type is None:
                layers.append(
                    Transformer(int(in_channels * (scale_factor ** (i + 1))), 8, 64, 4)
                )
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        out = self.in_block(x)
        for layer in self.downsample_block:
            out = layer(out)
        for layer in self.middle_block1:
            out = layer(out)
        for layer in self.middle_block2:
            if isinstance(layer, ResidualBlock):
                out = layer(out, s)
            else:
                out = layer(out)
        for layer in self.upsample_block:
            if isinstance(layer, ResidualBlock):
                out = layer(out, s)
            else:
                out = layer(out)
        out = self.out_block(out)
        return out
