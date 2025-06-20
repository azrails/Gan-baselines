"""
Template module, may used without changes
"""

import json
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parent.parent.parent


def read_json(fname: Path | str) -> dict:
    """
    Reads json file

    Args:
        fname (Path | str): path to json file

    Returns:
        dict: dict representaition of json
    """
    fname = Path(fname)
    with fname.open("rt") as f:
        return json.load(f)
