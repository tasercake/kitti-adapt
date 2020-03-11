#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import pytorch_lightning as pl


class VkittiImageDataSet(Dataset):
    def __init__(self, vkitti_dir, subsets=("rgb",), transforms=(None,)):
        """
        Args:
            transforms (tuple): Sequence of torch transforms corresponding to each subset
        """
        if not len(subsets):
            raise ValueError("Must provide at least one VKITTI subset to load.")

        self.vkitti_dir = Path(vkitti_dir)
        self.subsets = subsets
        self.transforms = transforms

        # Load filenames and check dataset integrity
        # TODO: convert these prints to logging statements
        path_templates = []
        for subset in self.subsets:
            templates = filter(lambda f: f.is_file, self.vkitti_dir.joinpath(subset).rglob(f"**/*"))  # Recursively get files in subset
            templates = map(lambda f: str(f).replace(subset, "{}"), templates)  # Replace subset name with templating brackets
            templates = set(map(lambda f: Path(f).with_suffix(""), templates))  # Remove extension
            path_templates.append(templates)
            print(f"Subset '{subset}' contains {len(templates)} files")
        if len(self.subsets) > 1:
            self.path_templates = path_templates[0].intersection(*path_templates[1:])
            if len(self.path_templates) != max(map(len, path_templates)):
                print("WARNING: Found inconsistencies in dataset.")
                print("The following is a squence of files found in at least one selected subset, but not found in at least one other subset.")
                for i in range(len(self.subsets)):
                    diff = list(map(str, path_templates[i].difference(self.path_templates)))
                    if diff:
                        print(f"'{self.subsets[i]}': {diff}")
        else:
            self.path_templates = path_templates[0]

        print(f"Found a total of {len(self.path_templates)} valid files.")

    def __getitem__(self, idx):
        return self.filepaths

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
        dataset = VkittiImageDataSet(self.vkitti_dir, (self.rgb_dirname, self.depth_dirname))
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader

    def val_dataloader(self):
        # transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        dataset = VkittiImageDataSet(self.vkitti_dir, (self.rgb_dirname, self.depth_dirname))
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def training_step(self, batch, batch_nb):
        x, y = batch
        print("TRAIN STEP")
        pred = self.forward(x)
        loss = F.mse_loss(pred, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        pred = self.forward(x)
        loss = F.mse_loss(pred, y)
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
