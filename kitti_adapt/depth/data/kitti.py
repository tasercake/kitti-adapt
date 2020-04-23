import os
from pathlib import Path
from glob import glob
from natsort import natsorted
import itertools

import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset

class KittiDepthDataSet(Dataset):
    def __init__(
        self,
        kitti_dir,
        scenes=None,
        transform=None,
        exclude=("image_00", "image_01", "image_03",),
    ):
        self.kitti_dir = kitti_dir
        self.scenes = scenes
        self.transform = transform
        self.exclude = exclude

    def get(self, idx):
        pass
