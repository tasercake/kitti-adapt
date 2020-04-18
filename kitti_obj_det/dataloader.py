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
	def __init__(self,images_dir, label_path,  device,transformation = None):


		# Find Root Directory then iterate trhough the 

		'''
		Initialise the relevant variables
		'''
		self.images = list(sorted(os.listdir(images_dir)))
		#print(images)
		self.images_dir = images_dir

		self.labels = list(sorted(os.listdir(label_path)))
		self.label_path = label_path
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


	def __getitem__(self, index):
		'''
		Laod images, masks, and get the bounding boxes
		'''

		image_path = os.path.join(self.images_dir,self.images[index])
		images = self._load_images(image_path)
		h, w = images.size

		bbox = os.path.join(self.label_path,self.images[index][:-3]+"txt")
		#bbox = os.path.join(self.label_path[index])
		df = pd.read_csv(bbox, sep = ' ',header=None)
		print('1')
		bboxdf = df.iloc[:,4:8]

		left = (np.array(bboxdf[4]) ) #/ h
		right = (np.array(bboxdf[6]) ) # / h
		top = (np.array(bboxdf[5]) )  #/ w
		bot = (np.array(bboxdf[7]) )  # / w

		'''
		Normalising Bounding Boxes
		'''
		boxes = list(np.stack([left, top,right, bot],axis=1))

		box_label = []
		print('2')

		labeldf = df[[0,4,5,6,7]]
		print('3')
		label_dict= { 'Car':1, 'Van':2, 'Truck':3,
		'Pedestrian':4, 'Person_sitting':5, 'Cyclist':6, 'Tram':7,
		'Misc':8, 'DontCare':9}

		labeldf['int_label'] = labeldf[0].map(label_dict)

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