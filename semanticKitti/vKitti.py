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

class vKittiDataset(Dataset):
    def __init__(self, img_dir,label_dir, transform=None):
        self.label_dir = label_dir
        # self.root_dir = "vKitti_RGB/Scene01/15-deg-left/frames/rgb/Camera_0/"
        self.transforms = transform
        self.image_filename = [str(file) for file in glob.glob(img_dir + "*.jpg")]

    def __len__(self):
        return len(self.image_filename)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        full_img_name = self.image_filename[idx]
        read_img = cv2.imread(full_img_name,-1)
        filename = full_img_name.split('_')[-1].split('.')[0]
        label_img = cv2.imread(self.label_dir + 'classgt_' + filename + '.png',-1)
        # obj_ids = np.unique(np.numpy(label_img))
        target = {}
        # Misc 80 80 80
        misc_img_copy = label_img.copy()
        misc_pixels_mask = np.all(label_img == [80, 80, 80], axis=-1)
        non_misc_pixels_mask = np.any(label_img != [80, 80, 80], axis=-1)
        misc_img_copy[misc_pixels_mask] = [255, 255, 255]
        misc_img_copy[non_misc_pixels_mask] = [0, 0, 0]
        target['misc_color'] = misc_img_copy

        # Truck 160 60 60
        truck_img_copy = label_img.copy()
        truck_pixels_mask = np.all(label_img == [160, 60, 60], axis=-1)
        non_truck_pixels_mask = np.any(label_img != [160, 60, 60], axis=-1)
        truck_img_copy[truck_pixels_mask] = [255, 255, 255]
        truck_img_copy[non_truck_pixels_mask] = [0, 0, 0]
        target['truck_check'] = truck_img_copy
        print(misc_img_copy[:,:,0].shape)
        print(truck_img_copy[:,:,0].shape)
        trans = transforms.Compose([transforms.ToTensor()])

        xxx = torch.cat((trans(misc_img_copy[:,:,0]),trans(truck_img_copy[:,:,0])))
        print(xxx.shape)
        # # Car 255 127 80
        # car_color = np.uint8([[[255, 127, 80]]])
        # car_check = (np.sum(label_img == car_color) > 500).astype(int)
        # target['car_check'] = car_check
        #
        # car_color = np.uint8([[[160, 60, 60]]])
        # truck_img_copy = truck_color.copy()
        # truck_pixels_mask = np.all(misc_color == [80, 80, 80], axis=-1)
        # non_truck_pixels_mask = np.any(misc_color != [80, 80, 80], axis=-1)
        # truck_img_copy[truck_pixels_mask] = [255, 255, 255]
        # truck_img_copy[non_truck_pixels_mask] = [0, 0, 0]
        # target['truck_check'] = truck_color
        #
        #
        # # Van 255 127 80
        # van_color = np.uint8([[[0, 139, 139]]])
        # van_check = (np.sum(label_img == van_color) > 500).astype(int)
        # target['van_check'] = van_check


        # sample = {'image': read_img, 'label': label_one_hot}
        #
        # if self.transform:
        #     sample = self.transform(sample)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225])])
        transform2 = transforms.Compose([transforms.ToTensor()])
        if self.transforms is not None:
            img = transform(read_img)
            # target = transform(target)

        return img, xxx


class DataLoader_vkitti(pl.LightningModule):
    def __init__(self, model, img_dir, lab_dir, batch_size, transform):
        super(DataLoader_vkitti,self).__init__()
        self.img_dir = img_dir
        self.label_dir = lab_dir
        self.batch_size = batch_size
        self.transform = transform
        self.feature_extractor = model
        self.feature_extractor.eval()

    def forward(self, x):
        x = self.feature_extractor(x)['out']
        return x

    def load_img(self):
        image_filename = [file for file in glob.glob(self.img_dir + "*.jpg")]
        # print(image_filename[0].split('\\')[-1].split('.')[0])
        # filename = image_filename[0].split('_')[-1].split('.')[0]
        # label_img = cv2.imread(self.label_dir + 'classgt_' + filename + '.png',-1)

        # Convert BGR to RGB, now you will see the color of 'frame' image
        # is displayed properly.
        # label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)

        # Car 255 127 80
        # car_color = np.uint8([[[255, 130, 0]]])
        # height,width,channel = label_img.shape
        # blank_image = np.zeros(shape=[height, width, channel], dtype=np.uint8)
        # car_imag = blank_image + car_color
        # print(label_img.shape)
        # print(car_imag.shape)
        # test_img1 = np.sum(label_img == car_color)
        # test_img2 = np.sum(label_img == car_imag)
        # print(label_img.shape)
        # print(test_img1)
        # print(test_img2)
        # res_and = cv2.bitwise_and(label_img,car_imag)
        # plt.figure(figsize=(20, 10))
        # plt.subplot(131), plt.imshow(label_img)
        # plt.subplot(132), plt.imshow(car_imag)
        # plt.subplot(133), plt.imshow(test_img)
        # plt.show()
        # transform = transforms.Compose([transforms.ToTensor()])
        # print(transform(test_img))
        # print(self.transform(label_img))
        # car_check = np.sum(label_img == car_color)
        # print(car_check)
        return vKittiDataset(self.img_dir,self.label_dir,self.transform)

    def prepare_data(self):
        # transform
        transform = transforms.Compose([transforms.ToTensor()])
        # self.mnist_train = MNIST(os.getcwd(), train = True, download = True, transform=transform)
        # self.mnist_test = MNIST(os.getcwd(), train = True, download = True, transform=transform)
        # self.mnist_train, self.mnist_val = random_split(self.mnist_train, [55000, 5000])
        # print(type(self.mnist_train))
        # print(type(self.mnist_test))
        # download
        self.kitti_train = self.load_img()
        # print(type(self.load_img()))
        # kitti_test = self.load_img(train=False, download=True, transform=transform)

        # train/val split
        # kitti_train, kitti_val = random_split(kitti_train, [55000, 5000])

        # assign to use in dataloaders
        # self.train_dataset = kitti_train
        # self.val_dataset = kitti_val
        # self.test_dataset = kitti_test
    #
    # # stuff here is done once at the very beginning of training
    # # before any distributed training starts
    #
    # # download stuff
    # # save to disk
    # # etc...

    def train_dataloader(self):
        return DataLoader(self.kitti_train,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.kitti_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.kitti_train, batch_size=self.batch_size)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, label = batch
        output = self.forward(x)
        numpy_pred = output.new_tensor(x).numpy()
        # preds = []

        # for i in range(len(numpy_pred)):
        #     pred_img = numpy_pred[i]
        #     preds_one_hot = []
        #     # Misc 80 80 80
        #     misc_color = np.uint8([[[80, 80, 80]]])
        #     misc_color = (np.sum(pred_img == misc_color) > 500).astype(int)
        #     preds_one_hot.append(misc_color)
        #
        #     # Truck 160 60 60
        #     truck_color = np.uint8([[[160, 60, 60]]])
        #     truck_check = (np.sum(pred_img == truck_color) > 500).astype(int)
        #     preds_one_hot.append(truck_check)
        #
        #     # Car 255 127 80
        #     car_color = np.uint8([[[255, 127, 80]]])
        #     car_check = (np.sum(pred_img == car_color) > 500).astype(int)
        #     preds_one_hot.append(car_check)
        #
        #     # Van 255 127 80
        #     van_color = np.uint8([[[0, 139, 139]]])
        #     van_check = (np.sum(pred_img == van_color) > 500).astype(int)
        #     preds_one_hot.append(van_check)
        #     preds.append(preds_one_hot)

        # preds = torch.IntTensor(preds)
        # print(preds.shape)
        # print(preds)
        # print(label.shape)
        # print(label)
        criterion = torch.nn.MSELoss()
        loss = criterion(output, label)
        return {'loss': loss}
        # return loss (also works)




def run():
    # if torch.cuda.is_available():
    #     dev = "cuda:0"
    # else:
    #     dev = "cpu"
    # device = torch.device(dev)


    batch_size = 2
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
    img_directory = "../data/vKitti_RGB/Scene01/15-deg-left/frames/rgb/Camera_0/"

    label_directory = "../data/VKitti_classSeg/Scene01/15-deg-left/frames/classSegmentation/Camera_0/"
    # label_directory = "vKitti_RGB/Scene18/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg"
    model = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=2, aux_loss=None)
    dataloader = DataLoader_vkitti(model, img_directory, label_directory, batch_size, transform)
    # print(dataloader)
    trainer = pl.Trainer(gpus=0)
    trainer.fit(dataloader)

if __name__ == '__main__':
    run()
