import logging
import random
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)


class HQDataset(BaseDataset):
    """
    Custom Dataset class for the CelebA dataset. Automatically downloads
    data and annotations from Google Drive if not found in the specified root_dir.
    """

    celeba_url = "https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=1"
    afhq_url = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1"

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str = "source",
        dataset: str = "celeba",
        *args,
        **kwargs,
    ):
        if data_dir is None:
            data_dir = ROOT_PATH / "datasets"
            data_dir.mkdir(exist_ok=True, parents=True)

        self.dataset_name = dataset
        data_dir = Path(data_dir)
        self._data_dir = data_dir

        self.split = split
        self.labels = []

        index = self._get_index()

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        data_path = data_dict["path"]
        data_object = self.load_object(data_path)
        data_label = data_dict["label"]

        instance_data = {"data_object": data_object, "labels": data_label}
        if self.split == "reference":
            reference_path = data_dict["reference"]
            reference_object = self.load_object(reference_path)
            instance_data["reference_object"] = reference_object

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def load_object(self, path: Path) -> torch.Tensor:
        return Image.open(path).convert("RGB")

    def _get_index(self) -> dict[str : str | int]:
        # downloading images
        download_url = None
        if self.dataset_name == "celeba":
            image_folder = self._data_dir / "celeba_hq"
            zip_file = self._data_dir / "celeba_hq.zip"
            download_url = self.celeba_url
        elif self.dataset_name == "afhq":
            image_folder = self._data_dir / "afhq_v2"
            zip_file = self._data_dir / "afhq_v2.zip"
            download_url = self.afhq_url
        else:
            raise ValueError(f"Unknown dataset name {self.dataset_name}...")

        if not image_folder.exists():
            if not zip_file.exists():
                logger.debug("=" * 10 + "Downloading data..." + "=" * 10)
                response = requests.get(download_url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                progress_bar = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading",
                )
                with open(zip_file, "wb") as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192)):
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            logger.debug("=" * 10 + f"Extracting {zip_file}..." + "=" * 10)
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall(self._data_dir)

        if self.split == "source" or self.split == "reference":
            image_folder = image_folder / "train"
        elif self.split == "val":
            if self.dataset_name == "afhq":
                image_folder = image_folder / "test"
            else:
                image_folder = image_folder / "val"
        else:
            raise KeyError("Unknown split parameter")

        logger.debug("=" * 10 + f"Creating split: {self.split} index..." + "=" * 10)
        domains = [label for label in sorted(image_folder.iterdir()) if label.is_dir()]
        domain_to_idx = {label: i for i, label in enumerate(domains)}
        index = []
        for domain_name in domains:
            domain_path = image_folder / domain_name
            imgs = list(fn for fn in sorted(domain_path.iterdir()) if fn.is_file())
            if self.split == "reference":
                imgs_pair = random.sample(imgs, len(imgs))

            for i in range(len(imgs)):
                if self.split != "reference":
                    index.append(
                        {
                            "path": domain_path / imgs[i],
                            "label": domain_to_idx[domain_name],
                        }
                    )
                else:
                    index.append(
                        {
                            "path": domain_path / imgs[i],
                            "reference": domain_path / imgs_pair[i],
                            "label": domain_to_idx[domain_name],
                        }
                    )
                self.labels.append(domain_to_idx[domain_name])
        return index

    def get_balanced_sampler(self):
        weights = 1.0 / np.bincount(self.labels)
        weights = weights[self.labels]
        return WeightedRandomSampler(weights, len(weights))
