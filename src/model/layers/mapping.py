import torch
from torch import nn

from src.model.layers.initialization_utils import init_fc_zero


class MappingNetwork(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        shared_size: int,
        unshared_size: int,
        domains: int,
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.in_layer = nn.Linear(latent_dim, hidden_dim)

        # shared domain params
        self.shared = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                for _ in range(shared_size)
            ]
        )

        # unshared params specific for each domain
        self.unshared = nn.ModuleList()
        for _ in range(domains):
            self.unshared.append(
                nn.Sequential(
                    *[
                        nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                        for _ in range(unshared_size - 1)
                    ],
                    nn.Linear(hidden_dim, embedding_dim)
                )
            )
        self.apply(init_fc_zero)

    def forward(self, z, y):
        out = self.activation(self.in_layer(z))
        out = self.shared(out)

        # list[batch_size  * embedding_dim]
        domain_styles = []
        for layer in self.unshared:
            domain_styles.append(layer(out))

        # batch_size * domains * embedding_dim
        domain_styles = torch.stack(domain_styles, dim=1)
        idx = torch.arange(y.size(0), device=y.device)
        out = domain_styles[idx, y]
        return out
