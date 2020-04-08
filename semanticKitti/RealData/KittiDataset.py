import glob
import cv2
import torch
from torchvision.transforms import transforms
import numpy as np
import os
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset




class vKittiDataset(Dataset):
    def __init__(self, img_dir,colors_dic, transform):
        def load_images_from_folder(folder):
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
        
        img_directory = "D:\\data_semantics\\training\\semantic_rgb"
        
        def check_dims(img_directory):
            images, attatched_filenames, filenames = load_images_from_folder(img_directory)
            sizes = []
            for img in images:
                sizes.append(img.shape)
            uSizes=set(sizes)
            sizeD = {}
            for size in uSizes:
                sizels = []
                for k,v in attatched_filenames:
                    if v.shape == size:
                        sizels.append(k)
                sizeD[size] = sizels
                        
            # print(uSizes)
            for thing in sizeD:
                print(thing)
                print(sizeD[thing])
            return sizeD,attatched_filenames
        
        def standardise_image_dims(img_directory):
            sizeD,attatched_filenames = check_dims(img_directory)
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
                newfile = cv2.resize(badfile,(selected_dims[1],selected_dims[0]), interpolation = cv2.INTER_AREA)
                print(badfile.shape,newfile.shape)
                attatched_filenames_d[badfilename] = newfile
            corrected_images = list(attatched_filenames_d.values())
            return attatched_filenames_d
#        self.label_dir = label_dir
        print("init kitti dataset")
        self.images, self.attatched_filenames,self.image_filename = load_images_from_folder(img_dir)
        self.images = standardise_image_dims(img_dir)
        print(len(self.image_filename))
        self.img_dir = img_dir
        # self.image_filename = [str(file) for file in glob.glob(img_dir)]
        # print(self.image_filename)
        # print(len(self.image_filename))
        # data_path = os.path.join(img_dir,'*g')
        # files = glob.glob(data_path)
        # data = []
        # for f1 in files:
        #     img = cv2.imread(f1)
        #     data.append(img)
        # print()
        # colorsls = []
        # counter = 0
        # for img in data:
        #     colors = set( tuple(v) for m2d in img for v in m2d )
        #     colorsls.append(colors)
        #     counter+=1
        #     print(counter)
        # allcolors = [i for sub in colorsls for i in sub]
        # uniquecolors = set(allcolors)
        # print(uniquecolors)
        # print(len(uniquecolors))
        self.colors_dic = colors_dic
        # self.allcolors = uniquecolors
        self.transforms = transform
        self.tensor_transform = transforms.Compose([transforms.ToTensor()])
        print(self.colors_dic)
    def __len__(self):
        return len(self.image_filename)

    def create_mask(self,cat_color,img_copy):
        pixels_mask = np.all(img_copy == cat_color, axis=-1)
        non_pixels_mask = np.any(img_copy != cat_color, axis=-1)
        img_copy[pixels_mask] = [255, 255, 255]
        img_copy[non_pixels_mask] = [0, 0, 0]
        return img_copy[:,:,0]

    def create_all_masks(self,img):
        cat_lst = self.colors_dic.keys()
        mask_lst = []
        # print(mask_lst)
        # print('making masks for image')
        for cat in cat_lst:
            # print(cat)
            mask_lst.append(self.tensor_transform(self.create_mask(self.colors_dic[cat],img.copy())))
        return torch.cat(mask_lst)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Reading RGB Image
        img_name = self.image_filename[idx]
        read_img = self.images[img_name]
        print(img_name,type(read_img))
        all_masks = self.create_all_masks(read_img)
        # print("all masks:" + str(len(all_masks)))
        
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