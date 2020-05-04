# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 01:11:19 2020

@author: Jen Yang
"""
import os
import cv2

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
    return corrected_images

corrected_images = standardise_image_dims(img_directory)
sizes = []
for img in corrected_images:
    sizes.append(img.shape)
uSizes=set(sizes)

print(uSizes)