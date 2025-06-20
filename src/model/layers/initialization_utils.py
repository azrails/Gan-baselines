from torch import nn


def init_conv2d(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(
            m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
        )
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def init_fc_zero(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(
            m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
        )
        nn.init.constant_(m.bias, 0.0)


def init_fc_one(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(
            m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
        )
        nn.init.constant_(m.bias, 1.0)
