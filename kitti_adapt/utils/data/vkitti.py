from pathlib import Path
from glob import glob
from natsort import natsorted

import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset


class VkittiImageDataSet(Dataset):
    def __init__(self, vkitti_dir, subsets=("rgb",), transforms: dict = None):
        """
        """
        if not len(subsets):
            raise ValueError("Must provide at least one VKITTI subset to load.")

        self.vkitti_dir = Path(vkitti_dir)
        self.subsets = subsets
        self.transforms = transforms or {}

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
                        print(f"'{self.subsets[i]}': {diff}. Total missing: {len(diff)}")
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
            sample[subset] = data
        return sample

    def _load_file(self, path, subset):
        """
        Your file loading logic here!
        """
        if subset == "rgb":
            data = Image.open(path)
        elif subset == "depth":
            data = Image.open(path)
#             data = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(
#                 np.float32
#             )
        if subset in self.transforms:
            data = self.transforms[subset](data)
        return data

    def __len__(self):
        return len(self.path_templates)
