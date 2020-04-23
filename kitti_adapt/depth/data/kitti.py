import os
from pathlib import Path
from glob import glob
from natsort import natsorted
import itertools
from itertools import chain

import numpy as np
from PIL import Image
from PIL import ImageFilter
import cv2

from torch.utils.data import Dataset

class KittiDepthDataset(Dataset):
    def __init__(
        self,
        kitti_dir,
        scenes=None,  # Unused for now
        transform=None,
        subset="train",
        exclude=("image_00", "image_01", "image_03",),  # Unused for now
    ):
        self.kitti_dir = Path(kitti_dir)
        self.scenes = scenes
        self.transform = transform
        self.exclude = exclude
        self.subset = subset

        self.rgb_dir = self.kitti_dir / "rgb" / self.subset
        self.depth_dir = self.kitti_dir / "depth" / self.subset

        self.rgb_scenes = [f for f in self.rgb_dir.glob("*")]
        self.depth_scenes = [f for f in self.depth_dir.glob("*")]
        self.rgb_files = list(chain.from_iterable(map(lambda f: f.glob("**/*.png"), self.rgb_scenes)))
        self.depth_files = list(chain.from_iterable(map(lambda f: f.glob("**/*.png"), self.depth_scenes)))

        self.rgb_filenames = {f.name for f in self.rgb_files}
        self.depth_filenames = {f.name for f in self.depth_files}
        filename_diff = self.rgb_filenames.symmetric_difference(self.depth_filenames)
        self.rgb_files = list(filter(lambda f: f.name not in filename_diff, self.rgb_files))
        self.depth_files = list(filter(lambda f: f.name not in filename_diff, self.depth_files))

        self.depth_uniques = {os.path.join(f.parents[3].name, f.name) for f in self.depth_files}
        self.rgb_uniques = {os.path.join(f.parents[2].name, f.name) for f in self.rgb_files}
        structure_diff = self.depth_uniques.symmetric_difference(self.rgb_uniques)
        filename_diff = {os.path.basename(f) for f in structure_diff}
        dirname_diff = {os.path.dirname(f) for f in structure_diff}
        self.rgb_files = [
            f for f in self.rgb_files if
            (f.name not in filename_diff) or (f.parents[2].name not in dirname_diff)
        ]
        self.depth_files = [
            f for f in self.depth_files if
            (f.name not in filename_diff) or (f.parents[3].name not in dirname_diff)
        ]
        assert len(self.depth_files) == len(self.rgb_files)

    def get(self, idx, transform=True):
        rgb_file = self.rgb_files[idx]
        depth_file = self.depth_files[idx]
        rgb = Image.open(rgb_file)
        depth = Image.open(depth_file)
        depth = depth.filter(ImageFilter.MaxFilter(9))
        depth = np.array(depth)
        depth[depth == 0] = 65535
        depth = Image.fromarray(depth)
        sample = {"rgb": rgb, "depth": depth}
        if transform and self.transform:
            sample = self.transform(sample)
        return sample

    def __getitem__(self, idx):
        return self.get(idx, transform=True)

    def __len__(self):
        return len(self.rgb_files)
