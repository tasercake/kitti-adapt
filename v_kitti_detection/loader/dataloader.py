'''
Custom Data Loader for CV Project
'''


import cv2
import torch

from torch.utils.data import Dataset
from torch.utils.data import random_split

import os

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot  as plt

class KITTILoader(Dataset):
	'''
	Dataset class to load the VKITII dataset
	'''
	def __init__(self,images_dir, label_path, info_path, device,transformation = None):


		# Find Root Directory then iterate trhough the 

		'''
		Initialise the relevant variables
		'''
		self.images = list(sorted(os.listdir(images_dir)))
		self.images_dir = images_dir

		self.label_path = label_path
		self.info_path = info_path
		# self.bboxdf , self.infodf = self._load_info(label_path, info_path)
		self.transformation = transformation
		self.device = device

	def __len__(self):
		length = len(self.images)
		return length



	def _load_images(self, image_name):
		'''
		Load image after getting the image from the files. Used for this class operation
		Inputs: the image name 
		Outputs: Image from the array
		'''
		image = Image.open(image_name).convert('RGB')
		image.load()
		image = np.array(image)
		if len(image.shape) == 2:
			image = np.expand_dims(image, 2)
			image = np.repeat(image, 3, 2)
		return Image.fromarray(image)


	def _resize(self, image, imsize):
		'''
		Resizing the image while keeping the ratio between the height and width. Used for this class
		This is simple crop method
		Inputs:
		- Image (the image file after being loaded )
		'''
		width, height = image.size
		scale = imsize / min(width,height)
		new_width = int(np.ceil(scale * width))
		new_height = int(np.ceil(scale * height))
		new_image = image.resize( (new_width, new_height))
		return new_image


	def _load_info (self, label_path, info_path):

		'''
		Function takes in label and info.txt path to be processed

		Inputs:
		- label_path : String (label)
		- info_path : string

		Outputs:
		- df : pandas dataframe
		- info : pandas datframe
		'''

		bbox = os.path.join(label_path)
		info = os.path.join(info_path)

		df = pd.read_csv(bbox, sep = ' ')
		df = df[(df['cameraID']==0)]
		df = df.drop(columns = ['number_pixels', 'truncation_ratio', 'occupancy_ratio', 'isMoving'])
		infodf = pd.read_csv(info, sep= ' ')

		# infodf['labels'] = infodf['model'] + '_' + infodf['color']
		infodf = infodf.drop(columns = ['model', 'color'])


		return df, infodf

	def __getitem__(self, index):
		'''
		Laod images, masks, and get the bounding boxes
		'''

		image_path = os.path.join(self.images_dir,self.images[index])
		# masks_path = os.path.join(self.masks_dir, self.masks[index])

		images = self._load_images(image_path)

		h, w = images.size
		# masks = self._load_images(masks_path) # image masks ( segmentation )
		# masks = np.array(masks)

		# masks = torch.as_tensor(masks, dtype=torch.uint8)

		bboxdf, infodf = self._load_info(self.label_path, self.info_path)

		bboxdf = bboxdf[(bboxdf['frame']==index)]

		'''
		Scalar bounding boxes
		'''

		left = (np.array(bboxdf['left']) ) #/ h
		right = (np.array(bboxdf['right']) ) # / h
		top = (np.array(bboxdf['top']) )  #/ w
		bot = (np.array(bboxdf['bottom']) )  # / w

		'''
		Normalising Bounding Boxes
		'''
		boxes = list(np.stack([left, top,right, bot],axis=1))

		box_label = []

		labeldf = pd.merge(bboxdf, infodf, on='trackID')

		label_dict= {'Car': 1, 'Van': 2, 'Truck': 3}

		labeldf['int_label'] = labeldf['label'].map(label_dict)



		boxes = torch.as_tensor(boxes, dtype=torch.float32).to(self.device)
		labels = torch.as_tensor(labeldf['int_label'], dtype=torch.int64).to(self.device)

		for box, label in list(zip(boxes, labels)):

			area = (box[3]-box[1]) * (box[2]-box[0])

			# if area <= 500: continue


			targets = dict()
			targets['labels'] = label

			targets['boxes'] = box
			targets['area'] = area
			box_label.append(targets)

		if self.transformation is not None:
			images = self.transformation(images)

		return {'images' : images, 'target' : box_label } 
		


