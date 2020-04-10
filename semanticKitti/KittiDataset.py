import glob
import cv2
import torch
from torchvision.transforms import transforms
import numpy as np
import os
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset


class KittiDataset(Dataset):
    def __init__(self, img_dir,label_dir,colors_dic, transform):
        """
        Segmented images are the labels that should be returned along side the original image (The ground truth)
        :param img_dir:
        :param label_dir:
        :param colors_dic:
        :param transform:
        """
#        self.label_dir = label_dir
        print("init kitti dataset")
        self.org_images, self.org_attatched_filenames,self.org_image_filename = self.load_images_from_folder(img_dir)
        self.org_images = self.standardise_image_dims(label_dir)

        self.seg_images, self.seg_attatched_filenames,self.seg_image_filename = self.load_images_from_folder(label_dir)
        self.seg_images = self.standardise_image_dims(label_dir)
        print(len(self.image_filename))
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.colors_dic = colors_dic
        self.transforms = transform
        self.tensor_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_filename)

    def load_images_from_folder(self,folder):
        images = []
        filenames = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if ".png" in filename:
                filenames.append(filename)
            if img is not None:
                if ".png" in filename:
                    images.append(img)
        attatched_filenames = list(zip(filenames,images))
        return images, attatched_filenames, filenames

    def check_dims(self,img_directory):
        """
        Checks dimension and returns a list of full file names
        :param img_directory:
        :return:
        """
        images, attatched_filenames, filenames = self.load_images_from_folder(img_directory)
        sizes = []
        for img in images:
            sizes.append(img.shape)
        uSizes = set(sizes)
        sizeD = {}
        for size in uSizes:
            sizels = []
            for k, v in attatched_filenames:
                if v.shape == size:
                    sizels.append(k)
            sizeD[size] = sizels

        # print(uSizes)
        for thing in sizeD:
            print(thing)
            print(sizeD[thing])
        return sizeD, attatched_filenames

    def standardise_image_dims(self,img_directory):
        sizeD, attatched_filenames = self.check_dims(img_directory)
        selected_dims = (375, 1242, 3)
        no = []
        for i in list(sizeD.keys()):
            if i != selected_dims:
                no.append(sizeD[i])
        no = [item for sublist in no for item in sublist]
        # attatched_filenames_d = {k,l for k,l in attatched_filenames}
        attatched_filenames_d = dict(attatched_filenames)
        for badfilename in no:
            badfile = attatched_filenames_d[badfilename]
            newfile = cv2.resize(badfile, (selected_dims[1], selected_dims[0]), interpolation=cv2.INTER_AREA)
            print(badfile.shape, newfile.shape)
            attatched_filenames_d[badfilename] = newfile
        corrected_images = list(attatched_filenames_d.values())
        return attatched_filenames_d

    def create_mask(self,cat_color,img_copy):
        pixels_mask = np.all(img_copy == cat_color, axis=-1)
        non_pixels_mask = np.any(img_copy != cat_color, axis=-1)
        img_copy[pixels_mask] = [255, 255, 255]
        img_copy[non_pixels_mask] = [0, 0, 0]
        return img_copy[:,:,0]

    def create_all_masks(self,img):
        cat_lst = self.colors_dic.keys()
        mask_lst = []
        for cat in cat_lst:
            # print(cat)
            mask_lst.append(self.tensor_transform(self.create_mask(self.colors_dic[cat],img.copy())))
        return torch.cat(mask_lst)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Reading RGB Image
        org_img_name = self.org_image_filename[idx]
        org_resized_img = self.org_images[org_img_name]

        # Reading RGB Image
        seg_img_name = self.seg_image_filename[idx]
        seg_resized_img = self.seg_images[seg_img_name]
        seg_all_masks = self.create_all_masks(seg_resized_img)


        if self.transforms is not None:
            img = self.transforms(org_resized_img)


        return img, seg_all_masks