'''
Updated DataLoader using cv2 instead of PIL

IC
'''


import cv2
import torch

from torch.utils.data import Dataset
from torch.utils.data import random_split

import os