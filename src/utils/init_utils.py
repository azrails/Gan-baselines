import logging
import os
import random
import secrets
import shutil
import string
from pathlib import Path

import numpy
import torch
from omegaconf import DictConfig, OmegaConf

from src.logger.logger import setup_logging
from src.utils.io_utils import ROOT_PATH


def set_random_seed(seed: int) -> None:
    """
    Setting random seed for RNG for all devices.

    Args:
        seed (int): random seed
    """
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = seed


def set_determinisic() -> None:
    """
    Setting deterministic behavior for cuda devices
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def set_worker_seed(worker_id: int) -> None:
    """
    Set seed for each dataloader worker.

    For more info, see https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        worker_id (int): worker id
    """
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_id(length: int = 8) -> str:
    """
    Generate a random base-36 string of `length` digits.

    Args:
        length (int): length of a string.
    Returns:
        str: base-36 string with an experiment id.
    """
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def resume_training(save_dir: Path) -> str:
    """
    Load run_id from config file

    Args:
        save_dir (Path): path to directory with saved config

    Returns:
        str: base-36 string with an experiment id.
    """
    config = OmegaConf.load(save_dir / "config.yaml")
    run_id = config.writer.run_id
    print(f"Resuming training from run {run_id}...")
    return run_id


def saving_init(save_dir: Path, config: DictConfig) -> None:
    """_summary_

    Args:
        save_dir (Path): path to log directory
        config (DictConfig): hydra expirement config

    Raises:
        ValueError: log directory already exist
    """
    run_id = None
    if save_dir.exists():
        if config.trainer.get("resume_from") is not None:
            run_id = resume_training(save_dir)
        elif config.trainer.override:
            print(f"Overriding save directory {save_dir}...")
            shutil.rmtree(save_dir)
        else:
            raise ValueError(
                "Save directory already exist. Set override=True or change path/name"
            )
    save_dir.mkdir(exist_ok=True, parents=True)

    if run_id is None:
        run_id = generate_id(config.writer.id_length)

    OmegaConf.set_struct(config, False)
    config.writer.run_id = run_id
    OmegaConf.set_struct(config, True)
    OmegaConf.save(config, save_dir / "config.yaml")


def setup_saving_and_logging(config: DictConfig) -> logging.Logger:
    """
    Initialize logger and writer logging directory

    Args:
        config (DictConfig): hydra expirement config

    Returns:
        logging.Logger: base python logger
    """
    save_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    saving_init(save_dir, config)

    if config.trainer.get("resume_from") is not None:
        setup_logging(save_dir, append=True)
    else:
        setup_logging(save_dir, append=False)
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    return logger
