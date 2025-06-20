"""
Template module, may used without changes
"""

import logging
import logging.config
from pathlib import Path

from src.utils.io_utils import ROOT_PATH, read_json


def setup_logging(
    save_dir: Path,
    log_config: Path | str | None = None,
    default_level: int | str = logging.INFO,
    append: bool = False,
) -> None:
    """
    Setup logging configuration

    Args:
        save_dir (Path): path to logging dir
        log_config (Path | str | None, optional): path to config logger file. Defaults to None.
        default_level (logging._Level, optional): default logging level. Defaults to logging.INFO.
        append (bool, optional): if True append to file instead of owerwrite. Defaults to False.
    """
    if log_config is None:
        log_config = ROOT_PATH / "src" / "logger" / "logger_config.json"
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])
        logging.config.dictConfig(config)
    else:
        print(f"Warning: logging configuration file is not found {log_config}...")
        logging.basicConfig(level=default_level, filemode="a" if append else "w")
