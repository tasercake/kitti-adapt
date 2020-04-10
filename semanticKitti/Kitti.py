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
from KittiDataloader import DataLoader_kitti
from KittiDataset import KittiDataset

def run():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    # img = cv2.imread('../data/VKitti_classSeg/Scene01/15-deg-left/frames/classSegmentation/Camera_0/classgt_00000.png',-1)
    # print(img.shape)
    # unique_arr = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    # if [0,199,255] in unique_arr:
    #     print('xxx')

    ## Processing vColors.txt which contains the rgb values for the segmented image
    f = open('vColors.txt','r')
    virtual_color_dic = {}
    virtual_num_class = 0
    desired_classes = ['Car', 'TrafficLight', 'TrafficSign', 'Pole', 'GuardRail', 'Vegetation', 'Terrain', 'Undefined', 'Sky', 'Road']
    for line in f:
        cat,r,g,b = line.split()
        for cat in desired_classes:
            virtual_color_dic[cat] = [r,g,b]
            virtual_num_class += 1

    ### Processing rColors_org.txt which contains the rgb values for the segmented image
    f = open('rColors.txt','r')
    real_color_dic = {}
    real_num_class = 0
    desired_classes = ['Car', 'TrafficLight', 'TrafficSign', 'Pole', 'GuardRail', 'Vegetation', 'Terrain', 'Undefined', 'Sky', 'Road']
    for line in f:
        cat,r,g,b = line.split()
        for cat in desired_classes:
            real_color_dic[cat] = [r,g,b]
            real_num_class += 1

    batch_size = 2
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
    model = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=real_num_class, aux_loss=None)

    print('Reading Virtual Data')
    vir_img_directory = "../data/vKitti_RGB/Scene01/15-deg-left/frames/rgb/Camera_1/"
    vir_label_directory = "../data/VKitti_classSeg/Scene01/15-deg-left/frames/classSegmentation/Camera_1/"
    virtual_kitti_dataset = KittiDataset(vir_img_directory, vir_label_directory,virtual_color_dic,transform)

    print('Reading Real Data')
    real_img_directory = "../data/data_semantics/training/image_2/"
    real_label_directory = "../data/data_semantics/training/semantic_rgb/"
    real_kitti_dataset = KittiDataset(real_img_directory, real_label_directory,real_color_dic,transform)

    print('Creating Dataloader')
    dataloader = DataLoader_kitti(real_kitti_dataset, virtual_kitti_dataset, model, batch_size,0.8,10)
    trainer = pl.Trainer(gpus=0)
    trainer.fit(dataloader)

if __name__ == '__main__':
    run()
