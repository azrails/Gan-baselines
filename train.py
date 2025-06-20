import os
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import (
    set_determinisic,
    set_random_seed,
    setup_saving_and_logging,
)

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config: DictConfig):
    """
    Main for trining. Create and initialize model, optimizer, scheduler, etc.
    Runs trainer to train and evaluate model.

    Args:
        config (DictConfig): hydra train config
    """
    if config.trainer.get("seed", None) is not None:
        set_random_seed(config.trainer.seed)
    if config.trainer.deterministic:
        set_determinisic()

    logger = setup_saving_and_logging(config)
    dict_config = OmegaConf.to_container(config)
    writer = instantiate(config.writer, logger, dict_config)

    if config.trainer.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders = get_dataloaders(config, device)

    # build model
    model = instantiate(config.model.model).to(device)
    logger.info(model)

    # build loss and metric
    loss_function = {
        loss_name: instantiate(config.loss_function[loss_name]).to(device)
        for loss_name in config.loss_function.keys()
    }

    metrics = instantiate(config.metrics)

    # build optimizer and lr scheduler
    g_params = list(filter(lambda p: p.requires_grad, model.generator.parameters()))
    e_params = list(filter(lambda p: p.requires_grad, model.style_encoder.parameters()))
    f_params = filter(lambda p: p.requires_grad, model.mapping_network.parameters())
    d_params = filter(lambda p: p.requires_grad, model.discriminator.parameters())
    optimizers = {}
    optimizers["g_optimizer"] = instantiate(
        config.optimizer.g_optimizer, params=g_params
    )
    optimizers["e_optimizer"] = instantiate(
        config.optimizer.e_optimizer, params=e_params
    )
    optimizers["f_optimizer"] = instantiate(
        config.optimizer.f_optimizer, params=f_params
    )
    optimizers["d_optimizer"] = instantiate(
        config.optimizer.d_optimizer, params=d_params
    )

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizers,
        config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
