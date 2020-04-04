import glob
import cv2
import torch
from torchvision.transforms import transforms
import numpy as np
import os
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset



class vKittiDataset(Dataset):
    def __init__(self, img_dir, transform):
#        self.label_dir = label_dir
        print("init kitti dataset")
        self.image_filename = [str(file) for file in glob.glob(img_dir + "*.png")]
        data_path = os.path.join(img_dir,'*g')
        files = glob.glob(data_path)
        data = []
        for f1 in files:
            img = cv2.imread(f1)
            data.append(img)
        colorsls = []
        counter = 0
        for img in data:
            colors = set( tuple(v) for m2d in img for v in m2d )
            colorsls.append(colors)
            counter+=1
            print(counter)
        allcolors = [i for sub in colorsls for i in sub]
        uniquecolors = set(allcolors)
        print(uniquecolors)
        print(len(uniquecolors))
        self.allcolors = uniquecolors
        self.transforms = transform
        self.tensor_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_filename)

    def create_mask(self,cat_color,img_copy):
        pixels_mask = np.all(img_copy == cat_color, axis=-1)
        non_pixels_mask = np.any(img_copy != cat_color, axis=-1)
        img_copy[pixels_mask] = [255, 255, 255]
        img_copy[non_pixels_mask] = [0, 0, 0]
        return img_copy[:,:,0]

    def create_all_masks(self,img):
#        cat_lst = self.colors_dic.keys()
        mask_lst = []
        for cat in self.allcolors:
            mask_lst.append(self.tensor_transform(self.create_mask(self.colors_dic[cat],img.copy())))
        return torch.cat(mask_lst)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Reading RGB Image
        img_name = self.image_filename[idx]
        read_img = cv2.imread(img_name,-1)
        all_masks = self.create_all_masks(read_img)
        
        # Reading corresponding seg image
#        filename = img_name.split('_')[-1].split('.')[0]
#        label_img = cv2.imread(self.label_dir + 'classgt_' + filename + '.png',-1)
#        label_mask = self.create_all_masks(label_img)


        # sample = {'image': read_img, 'label': label_one_hot}
        #
        # if self.transform:
        #     sample = self.transform(sample)
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                                                             std=[0.229, 0.224, 0.225])])
        if self.transforms is not None:
            img = self.transforms(read_img)


        return img, all_masks