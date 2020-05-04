import cv2
import torch
from torchvision.transforms import transforms
import numpy as np
import os
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset


class KittiDataset(Dataset):
    def __init__(self, img_dir, label_dir, colors_dic, transform):
        """
        Segmented images are the labels that should be returned along side the original image (The ground truth)
        :param img_dir:
        :param label_dir:
        :param colors_dic:
        :param transform:
        """
        self.img_dir = img_dir
        self.org_image_filename = self.load_names_from_folder(img_dir)
        self.seg_image_filename = self.load_names_from_folder(label_dir)

        self.img_dir = img_dir
        self.label_dir = label_dir

        self.colors_dic = colors_dic
        self.transforms = transform
        self.tensor_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.org_image_filename)

    def load_names_from_folder(self, folder):
        filenames = []
        len_root = len(folder)
        if type(folder) == tuple:
            folder = folder[0]
        for root, dirs, files in os.walk(folder):
            #             print('root dir ', root)
            if 'data_semantics' in root or 'clone' in root:
                for filename in files:
                    if ".png" or ".jpg" in filename:
                        filenames.append(os.path.join(root, filename))
        return filenames

    def check_dims(self, img_directory):
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

        return sizeD, attatched_filenames

    def standardise_images(self, img):
        selected_dims = (375, 1242, 3)
        if img.shape != selected_dims:
            resized_file = cv2.resize(img, (selected_dims[1], selected_dims[0]), interpolation=cv2.INTER_AREA)
        else:
            return img
        return resized_file

    def create_mask(self, cat_color, img_copy):
        pixels_mask = np.all(img_copy == cat_color, axis=-1)
        non_pixels_mask = np.any(img_copy != cat_color, axis=-1)
        img_copy[pixels_mask] = [255, 255, 255]
        img_copy[non_pixels_mask] = [0, 0, 0]
        return img_copy[:, :, 0]

    def create_all_masks(self, img):
        cat_lst = self.colors_dic.keys()
        mask_lst = []
        for cat in cat_lst:
            # print(cat)
            mask_lst.append(self.tensor_transform(self.create_mask(self.colors_dic[cat], img.copy())))
        return torch.cat(mask_lst)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Reading RGB Image
        org_img_name = self.org_image_filename[idx]
        org_img = cv2.imread(org_img_name)
        org_resized_img = self.standardise_images(org_img)
        # Reading class segmentated Image
        seg_img_name = self.seg_image_filename[idx]
        seg_img = cv2.imread(seg_img_name)
        seg_resized_img = self.standardise_images(seg_img)
        seg_mask = self.create_all_masks(seg_resized_img)

        if self.transforms is not None:
            img = self.transforms(org_resized_img)

        return img, seg_mask