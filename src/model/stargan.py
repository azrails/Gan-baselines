import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class StarGAN(nn.Module):
    """_summary_
    StarGAN-V2 implementaition based on https://arxiv.org/abs/1912.01865
    """

    def __init__(
        self, spectr_norm, generator, discriminator, mapping_network, style_encoder
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.mapping_network = mapping_network
        self.style_encoder = style_encoder
        if spectr_norm:
            self.apply_spectral_norm(self.discriminator)

    def apply_spectral_norm(self, module):
        for _, layer in module.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                spectral_norm(layer)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """

        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
