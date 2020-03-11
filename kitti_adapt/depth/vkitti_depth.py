#!/usr/bin/env python3
from pathlib import Path
from glob import glob
from natsort import natsorted

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import pytorch_lightning as pl


class VkittiImageDataSet(Dataset):
    def __init__(self, vkitti_dir, subsets=("rgb",), transform: dict = None):
        """
        Args:
        """
        if not len(subsets):
            raise ValueError("Must provide at least one VKITTI subset to load.")

        self.vkitti_dir = Path(vkitti_dir)
        self.subsets = subsets
        self.transform = transform or {}

        # Load filenames and check dataset integrity
        # TODO: convert these prints to logging statements
        path_templates = []
        for subset in self.subsets:
            templates = {
                f
                for f in self.vkitti_dir.joinpath(subset).rglob(f"**/*")
                if f.is_file()
            }  # Recursively get files in subset
            templates = map(
                lambda f: str(f).replace(subset, "{subset}"), templates
            )  # Replace subset name with templating brackets
            templates = set(
                map(lambda f: Path(f).with_suffix(""), templates)
            )  # Remove extension
            path_templates.append(templates)
            print(f"Subset '{subset}' contains {len(templates)} files")
        if len(self.subsets) > 1:
            self.path_templates = path_templates[0].intersection(*path_templates[1:])
            if len(self.path_templates) != max(map(len, path_templates)):
                print("WARNING: Found inconsistencies in dataset.")
                print(
                    "The following is a squence of files found in at least one selected subset, but not found in at least one other subset."
                )
                for i in range(len(self.subsets)):
                    diff = list(
                        map(str, path_templates[i].difference(self.path_templates))
                    )
                    if diff:
                        print(f"'{self.subsets[i]}': {diff}")
        else:
            self.path_templates = path_templates[0]
        self.path_templates = natsorted(self.path_templates)
        print(f"Found a total of {len(self.path_templates)} valid files.")

    def __getitem__(self, idx):
        template = self.path_templates[idx]
        sample = {}
        for subset in self.subsets:
            path = str(template).format(subset=subset)
            path = glob(path + "*")
            if not path:
                raise ValueError(f"Could not find file matching '{path}'.")
            data = self._load_file(path[0], subset)
            print(data.shape)
            sample[subset] = data
        return sample

    def _load_file(self, path, subset):
        """
        Your file loading + pre-processing logic here!
        """
        if subset == "rgb":
            print("LOADING RGB")
            return cv2.imread(path)
        elif subset == "depth":
            print("LOADING DEPTH")
            return cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(
                np.float
            )

    def __len__(self):
        return 12


class DepthEstimator(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        # Define data paths
        self.vkitti_dir = self.vkitti_dir
        self.rgb_dirname = "rgb"
        self.depth_dirname = "depth"

        self.batch_size = 1
        self.learning_rate = 1e-5

        self.stem = torchvision.models.segmentation.fcn_resnet50(progress=True)

    def forward(self, x):
        return self.stem(x)

    def train_dataloader(self):
        # transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        dataset = VkittiImageDataSet(
            self.vkitti_dir, (self.rgb_dirname, self.depth_dirname)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader

    def val_dataloader(self):
        # transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        dataset = VkittiImageDataSet(
            self.vkitti_dir, (self.rgb_dirname, self.depth_dirname)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def training_step(self, batch, batch_nb):
        rgb = batch["rgb"]
        depth = batch["depth"]
        pred = self.forward(rgb)
        loss = F.mse_loss(pred, depth)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        rgb = batch["rgb"]
        depth = batch["depth"]
        pred = self.forward(rgb)
        loss = F.mse_loss(pred, depth)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        # avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        # tensorboard_logs = {"val_loss": avg_loss}
        # return {"avg_loss": avg_loss, "log": tensorboard_logs}
        return {}


if __name__ == "__main__":
    VKITTI_DIR = "data/vkitti"

    model = DepthEstimator(vkitti_dir=VKITTI_DIR,)
    trainer = pl.Trainer()
    trainer.fit(model)
