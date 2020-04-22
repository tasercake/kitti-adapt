import os
from pathlib import Path
from glob import glob
from natsort import natsorted
import itertools

import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset


class VkittiImageDataSet(Dataset):
    def __init__(self, vkitti_dir, subsets=("rgb",), transforms: dict = None, limit=None, split=False):
        """
        """
        if not len(subsets):
            raise ValueError("Must provide at least one VKITTI subset to load.")

        self.vkitti_dir = Path(vkitti_dir)
        self.subsets = subsets
        self.transforms = transforms or {}
        self.limit = limit
        self.split = split

        # Load filenames and check dataset integrity
        # TODO: convert these prints to logging statements
        path_templates = []
        for subset in self.subsets:
            # Recursively get files in subset
            templates = [f for f in glob(
                os.path.join(str(self.vkitti_dir), subset, "**/*"),
                recursive=True,
            )]
            # Filter files only
            templates = filter(os.path.isfile, templates)
            if self.limit:
                templates = itertools.islice(templates, self.limit)
            # Replace subset name with templating brackets
            templates = map(lambda f: f.replace(subset, "{subset}"), templates)
            # Remove extension
            templates = map(lambda f: Path(f).with_suffix(""), templates)
            # Iterate over generator and create concrete set
            templates = set(templates)
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
                        print(f"'{self.subsets[i]}': {diff}. Total missing: {len(diff)}")
        else:
            self.path_templates = path_templates[0]
        self.path_templates = natsorted(self.path_templates)
        print(f"Found a total of {len(self.path_templates)} valid files.")

    def get(self, idx, transform=False):
        template = self.path_templates[idx]
        sample = {}
        for subset in self.subsets:
            path = str(template).format(subset=subset)
            path = glob(path + "*")
            if not path:
                raise ValueError(f"Could not find file matching '{path}'.")
            data = self._load_file(path[0], subset)
            if transform and subset in self.transforms:
                data = self.transforms[subset](data)
            sample[subset] = data
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
