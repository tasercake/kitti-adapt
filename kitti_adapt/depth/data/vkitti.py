import os
from pathlib import Path
from glob import glob
from natsort import natsorted
import itertools

import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset


class VkittiImageDataset(Dataset):
    def __init__(
        self,
        vkitti_dir,
        scenes=None,
        subsets=("rgb",),
        transform=None,
        single_camera=True,
        exclude=("clone", "30-deg-right", "30-deg-left", "15-deg-right"),
    ):
        """
        """
        if not len(subsets):
            raise ValueError("Must provide at least one VKITTI subset to load.")

        self.vkitti_dir = Path(vkitti_dir)
        self.scenes = scenes
        self.subsets = subsets
        self.transform = transform
        self.single_camera = single_camera
        self.exclude = exclude

        # Load filenames and check dataset integrity
        # TODO: convert these prints to logging statements
        path_templates = []
        for subset in self.subsets:
            # Recursively get files in subset
            templates = [
                f
                for f in glob(
                    os.path.join(str(self.vkitti_dir), subset, "**/*"), recursive=True,
                )
            ]
            # Filter files only
            templates = filter(os.path.isfile, templates)
            if self.scenes:
                templates = filter(lambda f: any(scn in f for scn in self.scenes), templates)
            if self.single_camera:
                templates = filter(lambda f: "Camera_0" in f, templates)
            if self.exclude:
                templates = filter(
                    lambda f: all(excl not in f for excl in self.exclude), templates
                )
            # Replace subset name with templating brackets
            templates = map(lambda f: f.replace(subset, "{subset}"), templates)
            # Remove extension
            templates = map(lambda f: Path(f).with_suffix(""), templates)
            # Iterate over generator and create concrete set
            templates = set(templates)
            path_templates.append(templates)
            # print(f"Subset contains {len(templates)} files")
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
                        print(
                            f"'{self.subsets[i]}': {diff}. Total missing: {len(diff)}"
                        )
        else:
            self.path_templates = path_templates[0]
        self.path_templates = natsorted(self.path_templates)
        print(f"Dataset contains {len(self.path_templates)} files.")

    def get(self, idx, transform=True):
        template = self.path_templates[idx]
        sample = {}
        for subset in self.subsets:
            path = str(template).format(subset=subset)
            path = glob(path + "*")
            if not path:
                raise ValueError(f"Could not find file matching '{path}'.")
            try:
                data = self._load_file(path[0], subset)
                sample[subset] = data
            except:
                print(f"Failed to load {path}")
                raise
        if transform and self.transform:
            sample = self.transform(sample)
        return sample

    def __getitem__(self, idx):
        return self.get(idx, transform=True)

    def _load_file(self, path, subset):
        """
        Your file loading logic here!
        """
        if subset == "rgb":
            data = Image.open(path)
        elif subset == "depth":
            data = Image.open(path)
        return data

    def __len__(self):
        return len(self.path_templates)
