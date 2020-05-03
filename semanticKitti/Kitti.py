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

    ## Processing vColors.txt which contains the rgb values for the segmented image
    f = open('vColors.txt', 'r')
    virtual_color_dic = {}
    virtual_num_class = 0
    #     desired_classes = ['Car', 'TrafficLight', 'TrafficSign', 'Pole', 'GuardRail', 'Vegetation', 'Terrain', 'Undefined', 'Sky', 'Road']
    desired_classes = ['Car', 'Undefined']
    for line in f:
        cat, r, g, b = line.split()
        if cat in desired_classes:
            virtual_color_dic[cat] = [r, g, b]
            virtual_num_class += 1

    ### Processing rColors_org.txt which contains the rgb values for the segmented image
    f = open('rColors.txt', 'r')
    real_color_dic = {}
    real_num_class = 0
    # desired_classes = ['Car', 'TrafficLight', 'TrafficSign', 'Pole', 'GuardRail', 'Vegetation', 'Terrain', 'Undefined', 'Sky', 'Road']
    desired_classes = ['Car', 'Undefined']
    for line in f:
        cat, r, g, b = line.split()
        if cat in desired_classes:
            real_color_dic[cat] = [r, g, b]
            real_num_class += 1

    batch_size = 1
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
    print('Number of classes', real_num_class)
    model = models.segmentation.fcn_resnet101(pretrained=True, progress=True, aux_loss=None)
    i = 0
    for child in model.children():
        i += 1
        #         print(i)
        #         print(child)
        # for param in child.parameters():
        #     print(param.requires_grad)
        print(i)
        if i < 3:
            print(i)
            for param in child.parameters():
                param.requires_grad = False

    print('Reading Virtual Data')
    vir_img_directory = '../data/vKitti_RGB',
    vir_label_directory = '../data/vKitti_classSeg'
    virtual_kitti_dataset = KittiDataset(vir_img_directory, vir_label_directory, virtual_color_dic, transform)
    print(len(virtual_kitti_dataset))
    print('Reading Real Data')
    real_img_directory = "../data/data_semantics/training/image_2"
    real_label_directory = "../data/data_semantics/training/semantic_rgb"
    real_kitti_dataset = KittiDataset(real_img_directory, real_label_directory, real_color_dic, transform)

    print('Creating Dataloader')
    # All real
    dataloader1 = DataLoader_kitti(real_kitti_dataset, virtual_kitti_dataset, model, batch_size, 0, real_num_class,
                                   0.005)
    # dataloader2 = DataLoader_kitti(real_kitti_dataset, virtual_kitti_dataset, model, batch_size, 0.25, real_num_class,
    #                                0.005)
    # dataloader3 = DataLoader_kitti(real_kitti_dataset, virtual_kitti_dataset, model, batch_size, 2, real_num_class,
    #                                0.0005)
    # dataloader4 = DataLoader_kitti(real_kitti_dataset, virtual_kitti_dataset, model, batch_size, 5, real_num_class,
    #                                0.0005)
    #     dataloader5 = DataLoader_kitti(real_kitti_dataset, virtual_kitti_dataset, model, batch_size,1,real_num_class,0.0005)

    print('---------------------------------------------------------break 1 - 100% Real----------------------------')
    trainer = pl.Trainer(min_epochs=0, max_epochs=0, gpus=1, early_stop_callback=True)
    trainer.fit(dataloader1)
    trainer.test()
    # print('---------------------------------------------------------break 2 - 75% Real----------------------------')
    # trainer = pl.Trainer(min_epochs=5, max_epochs=10, gpus=1, early_stop_callback=True)
    # trainer.fit(dataloader2)
    # trainer.test()
    # print('---------------------------------------------------------break 3 - 50% Real----------------------------')
    # trainer = pl.Trainer(min_epochs=5, max_epochs=10, gpus=1, early_stop_callback=True)
    # trainer.fit(dataloader3)
    # trainer.test()
    # print('---------------------------------------------------------break 4 - 25% Real----------------------------')
    # trainer = pl.Trainer(min_epochs=5, max_epochs=10, gpus=1, early_stop_callback=True)
    # trainer.fit(dataloader4)
    # trainer.test()
    # #     print('---------------------------------------------------------break 5 - 0% Real----------------------------')
    # #     trainer = pl.Trainer(min_epochs=5,max_epochs=10,gpus=1,early_stop_callback=True)
    # #     trainer.fit(dataloader5)
    # #     trainer.test()
    #
    # #     trainer = pl.Trainer(min_epochs=5,max_epochs=5,gpus=1)
    # #     trainer.fit(dataloader6)
    # #     trainer.test()
    # #     print('---------------------------------------------------------break 1 - 50% Real  50% Virtual----------------------------')

    print('I am done')


if __name__ == '__main__':
    run()
