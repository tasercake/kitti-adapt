import xml.etree.ElementTree as ET
import glob
import cv2
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt  # patch-wise similarities, droi images
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from torch.optim import Adam
import os
from skimage import io, transform
import torch.nn.functional as F
from KittiDataloader import DataLoader_vkitti
from KittiDataset import vKittiDataset

def run():
    # if torch.cuda.is_available():
    #     dev = "cuda:0"
    # else:
    #     dev = "cpu"
    # device = torch.device(dev)
    # img = cv2.imread('../data/VKitti_classSeg/Scene01/15-deg-left/frames/classSegmentation/Camera_0/classgt_00000.png',-1)
    # print(img.shape)
    # unique_arr = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    # if [0,199,255] in unique_arr:
    #     print('xxx')
    f = open('colors.txt','r')
    color_dic = {}
    num_class = 0
    desired_classes = ['Car', 'TrafficLight', 'TrafficSign', 'Pole', 'GuardRail', 'Vegetation', 'Terrain', 'Undefined', 'Sky', 'Road']
    for line in f:
        cat,r,g,b = line.split()
        for cat in desired_classes:
            color_dic[cat] = [r,g,b]
            num_class += 1
    print(num_class)
    batch_size = 2
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
    img_directory = "../data/vKitti_RGB/Scene01/15-deg-left/frames/rgb/Camera_0/test/"
    label_directory = "../data/VKitti_classSeg/Scene01/15-deg-left/frames/classSegmentation/Camera_0/test/"
    model = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=16, aux_loss=None)
    kitti_dataset = vKittiDataset(img_directory, label_directory,color_dic,transform)
    dataloader = DataLoader_vkitti(kitti_dataset, model, batch_size)
    trainer = pl.Trainer(gpus=0)
    trainer.fit(dataloader)

if __name__ == '__main__':
    run()
