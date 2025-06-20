from src.loss.cycle import CycleLoss
from src.loss.gan_loss import AdversarialLoss
from src.loss.loss import DiscriminatorLoss, GeneratorLoss
from src.loss.r1_term import R1Regulaizer
from src.loss.relativistic_loss import (
    RelativisticDiscriminatorLoss,
    RelativisticGeneratorLoss,
)
from src.loss.style_diversity import StyleDiversityLoss
from src.loss.style_reconstruction import StyleRecontructionLoss
from src.loss.wasserstein import WassersteinAdversarialLoss
