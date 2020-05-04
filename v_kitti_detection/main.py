'''
CV Virtual KITTI object detection

Ivan Christian & Billio Jeverson
'''

from loader.dataloader import KITTILoader
from loader.realloader import REALLoader

# from utils.vis import save_image

import cv2
import numpy as np
import math
from tqdm import tqdm
from pprint import pprint


import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist


from torch.utils.data import DataLoader, random_split, ConcatDataset
import os
import torch
from torchvision import transforms
import torchvision.models as models

import torchvision

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont


def is_dist_avail_and_initialized():
	'''
	Helper Function
	'''
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
	'''
	Helper Function
	'''
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()

def reduce_dict(input_dict, average=True):
	"""
	Args:
		input_dict (dict): all the values will be reduced
		average (bool): whether to do average or sum
	Reduce the values in the dictionary from all processes so that all processes
	have the averaged results. Returns a dict with the same fields as
	input_dict, after reduction.
	"""
	world_size = get_world_size()
	if world_size < 2:
		return input_dict
	with torch.no_grad():
		names = []
		values = []
		# sort the keys so that they are consistent across processes
		for k in sorted(input_dict.keys()):
			names.append(k)
			values.append(input_dict[k])
		values = torch.stack(values, dim=0)
		dist.all_reduce(values)
		if average:
			values /= world_size
		reduced_dict = {k: v for k, v in zip(names, values)}
	return reduced_dict




def move_to(obj, device):
	'''
	Function to move the dictionary to cuda
	'''
	if torch.is_tensor(obj):
		return obj.to(device)
	elif isinstance(obj, dict):
		res = {}
		for k, v in obj.items():
			res[k] = move_to(v, device)
		return res
	elif isinstance(obj, list):
		res = []
		for v in obj:
			res.append(move_to(v, device))
		return res
	else:
		raise TypeError("Invalid type for move_to")





def train( model, device , train_loader , optimizer, epochs, loss_func ):
	'''
	Function train is a function to train the model
	input: 
	- model : model
	- device : device that is being used
	- train_loader : data loader for training
	- optimizer : which optimizer used
	- epoch : at what epoch this is runnning in
	- loss_func : loss function used
	Output:
	- train_loss : float ( average train loss )
	'''

	model.train().to(device)

	train_losses = []

	train_loss = 0

	idx = 0 




	with tqdm(total = len(train_loader)) as bar:

		for index, batch in enumerate(train_loader):

			data = batch['images'].to(device)

			targets = batch['target']

			targets = move_to(targets, device)

			optimizer.zero_grad()

			loss_dict = model(data, targets) #outputs losses instead of predicition

			'''
			Epoch: [0]  [30/60]  eta: 0:00:11  lr: 0.002629  loss: 0.4705 (0.8935)  loss_classifier: 0.0991 (0.2517)  loss_box_reg: 0.1578 (0.1957)  loss_mask: 0.1970 (0.4204)  loss_objectness: 0.0061 (0.0140)  loss_rpn_box_reg: 0.0075 (0.0118)  time: 0.3403  data: 0.0044  max mem: 5081
Epoch: [0]  [40/60]  eta: 0:00:07  lr: 0.003476  loss: 0.3901 (0.7568)  loss_classifier: 0.0648 
			'''

			losses = sum(loss for loss in loss_dict.values())

			# reduce losses over all GPUs for logging purposes
			loss_dict_reduced = reduce_dict(loss_dict)
			losses_reduced = sum(loss for loss in loss_dict_reduced.values())

			loss_value = losses_reduced.item()

			optimizer.zero_grad()
			losses.backward()
			optimizer.step()

			train_loss += loss_value

			bar.update()

			idx = index


		train_loss /= (idx+1)
		# print(idx + 1)
	
	print(f'\nEpoch {epochs} Train loss : {train_loss}')

	return train_loss



def validate ( model , device , val_loader , loss_func, epochs):
	'''
	Function validate is used to check accuracy against the validation set
	
	Input:
	- model: model
	- device: string ( 'cuda' or 'cpu')
	- val_loader: DataLoader (validation data loader)
	- loss_func: (loss function chosen)

	Output:
	- val_loss: float (validation loss)
	- accuracy: float (validation accuracy)
	'''
	def match_size_calc_acc(t_target, t_pred):
		if t_target.shape[0] > t_pred.shape[0]:
			t_pred = t_pred.tolist()
			for i in range(len(t_target) - len(t_pred)):
				
				t_pred.append(0)
			t_pred = torch.tensor(t_pred)
		elif t_target.shape[0] < t_pred.shape[0]:
			t_target = t_target.tolist()
			for i in range(len(t_pred) - len(t_target)):
				t_target.append(0)
			t_target = torch.tensor(t_target)

		return t_pred, t_target

	model.eval().to(device)
	val_loss = 0
	val_accuracy = 0

	val_dicts= []
	print('Start Validation')

	with tqdm(total = len(val_loader)) as bar:

		with torch.no_grad():
			for idx , batch in enumerate(val_loader):
				data = batch['images'].to(device)
				targets = batch['target']
				targets = move_to(targets, device)
				prediction = model(data, targets) #outputs prediction instead of losses in eval mode.
				output = prediction[0]['scores']

				mean_score = output.mean()

				if torch.isnan(mean_score).any():
					mean_score = 0
				val_accuracy += mean_score

				bar.update()


	mean_val_accuracy  = val_accuracy / len(val_loader)

	print(f'Epoch {epochs} Validation Accuracy = {mean_val_accuracy}')
	return mean_val_accuracy


def load_image ( name , device ='cuda'):
	'''
	Function is used to load test images ( taken from the validation set ) to tensor for input. Helper function

	Input : 
	- path : string (root path for dataset)
	- name : string (image names)
	- transformation : Transform ( Transformation for the data --> Augmentation)
	- device :
	Output :
	- images : float tensor in cuda/device
	'''

	# preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
	preprocess = transforms.ToTensor()
	images = Image.open(name)
	images_tensor = preprocess(images).float()

	images_tensor = images_tensor.unsqueeze(0)

	return images_tensor.to(device), images

def test_visualisation(model_path, models, test_images_paths, test_dataset, mode ,device):
	'''
	Test and save sample images with bounding boxes and the labels

	Inputs:
	- model_path : string( path to the model )
	- models : weights (model weights)
	- test_images_paths : string ( model path )
	- test_dataset : Dataset (testing dataset)
	- device : device (cuda)

	Outputs:

	saved images
	'''
	import random

	label_dict= { 1: 'Car', 2 : 'Van' , 3 : 'Truck', 0 : 'Background'}

	directory = os.path.join('results', 'sample_bbox')


	def save_image(image,directory, filename):
		if not os.path.exists(directory):
			os.makedirs(directory)

		img_file = os.path.join(directory, filename)

		image.save(img_file)

	print(f'Loading model to score images. Scores saved {model_path}') # Testing the model

	mean_test_accuracy = []


	model_file = torch.load(model_path)
	models.load_state_dict(model_file)

	print('Model loaded')


	models.to(device)
	models.eval()

	with torch.no_grad():

		for path in test_images_paths:
			with tqdm(total = len(test_dataset)) as bar:
				for image in test_dataset.images:
					img = os.path.join(path, image)
					l_img, pic_image = load_image(img)

					output = models(l_img)

					im_show = l_img.permute(2,0,3,1)
					im_show = im_show.squeeze(1)

					labels = output[0]['labels']
					scores = output[0]['scores']

					mean_score = scores.mean()


					if torch.isnan(mean_score).any():
						continue

					# print(f'This is the mean score : {mean_score}')

					rect = output[0]['boxes']

					if rect.nelement() != 0:
						i = 0
						int_rects = rect.int().cpu().numpy()
						labels = labels.int().cpu().numpy()
						scores = scores.float().cpu().numpy()

						for int_rect, label, score in zip(int_rects, labels, scores):
							# print(label_dict[label], score)
							if score >= 0.5:
								r = random.randint(20,255)
								g = random.randint(20,255)
								b = random.randint(20,255)
								rgb = (r,g,b)

								x0,y0 ,x1,y1 = int_rect
								img1 = ImageDraw.Draw( pic_image )   
								font = ImageFont.truetype("bevan.ttf", 20)
								# img1.text([x0,y0,x1,y1+10], label, fill=(255,255,0))
								img1.text((0,0+i),f'{label_dict[label]} {score} ', rgb,font=font)
								img1.rectangle([x0,y0 ,x1,y1], outline = rgb, width = 3) # Draw the text on the 
								i += 20
							else:
								continue

						save_image(pic_image, os.path.join(directory, str(mode)), f'{image[:-4]}_samplebbox.png')

					mean_score = mean_score.float().cpu().numpy()

					mean_test_accuracy.append(mean_score)
					bar.update()

		print('FINISHED TESTING')



def custom_training(train_loader, val_loader, test_dataset,test_images_paths, mode, device= 'cuda', epochs=4):
	'''
	Custom Training function

	Inputs:
	- train_loader : 
	- val_loader :
	- test_dataset : 
	- test_images_path : 
	'''

	# load a pre-trained model for classification and return
	# only the features
	# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
	# # FasterRCNN needs to know the number of
	# # output channels in a backbone. For mobilenet_v2, it's 1280
	# # so we need to add it here
	# backbone.out_channels = 1280

	# # let's make the RPN generate 5 x 3 anchors per spatial
	# # location, with 5 different sizes and 3 different aspect
	# # ratios. We have a Tuple[Tuple[int]] because each feature
	# # map could potentially have different sizes and
	# # aspect ratios
	# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
	#                                    aspect_ratios=((0.5, 1.0, 2.0),))

	# # let's define what are the feature maps that we will
	# # use to perform the region of interest cropping, as well as
	# # the size of the crop after rescaling.
	# # if your backbone returns a Tensor, featmap_names is expected to
	# # be [0]. More generally, the backbone should return an
	# # OrderedDict[Tensor], and in featmap_names you can choose which
	# # feature maps to use.
	# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
	#                                                 output_size=7,
	#                                                 sampling_ratio=2)

	# # put the pieces together inside a FasterRCNN model
	# model = FasterRCNN(backbone,
	#                    num_classes=4,
	#                    rpn_anchor_generator=anchor_generator,
	#                    box_roi_pool=roi_pooler)
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

	num_classes = 4

	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
	loss_func = nn.BCEWithLogitsLoss().to(device)


	train_losses_vis = []
	val_losses_vis = []
	val_accuracy_list = []

	model_name = f'object_detect_model{mode}.pt'


	if os.path.exists(model_name):

		print('Weights exists')

		test_visualisation(model_name, model, test_images_paths,test_dataset, mode,device)


	else:

		print('Starting Training')



		for epoch in range(1, epochs + 1):
			train_loss = train(model, device, train_loader, optimizer, epoch, loss_func)
			accuracy = validate(model, device, val_loader, loss_func, epoch)

			if (len(train_losses_vis) > 0) and (train_loss < min(train_losses_vis)):
				torch.save(model.state_dict(), model_name)

			train_losses_vis.append(train_loss)

			val_accuracy_list.append(accuracy)




def get_labels_bbox(paths):
	new_paths_info = []
	new_paths_bbox = []
	for path in paths:
		info_path = os.path.join('labels', path, 'info.txt')
		bbox_path = os.path.join('labels', path, 'bbox.txt')
		new_paths_info.append(info_path)
		new_paths_bbox.append(bbox_path)


	return new_paths_info, new_paths_bbox


def get_image(paths):
	'''
	Function to get the path lists for the images

	INput:
	- paths : list (path lists)
	'''

	camera_list = ['Camera_0']

	new_list = []
	new_path_list = []
	for path in paths:
		for camera in camera_list:
			new_path = os.path.join(path, 'frames', 'rgb',camera)
			new_list.append(new_path)

	for new_path in new_list:
		new_path_image = os.path.join('images', new_path)

		new_path_list.append(new_path_image)


	return new_path_list

def get_real_image(paths):
	'''
	Function to get the path lists for the images

	INput:
	- paths : list (path lists)
	'''
	return glob.glob(paths+"/*.png")

def create_dataset(image_paths, labels_path, info_paths, device):
	'''
	Function to create dataset from multiple sources

	Inputs:
	- image_paths : list of the image paths
	- labels_path : list of the labels paths (in this case is the bounding boxes)
	- 
	'''

	preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

	datasetls = [] 
	for image_path, label_path, info_path in zip(image_paths, labels_path, info_paths):

		dataset = KITTILoader(image_path, label_path, info_path , device, transformation = preprocess)

		datasetls.append(dataset)

	all_dataset = ConcatDataset(datasetls)


	return all_dataset

def create_real_dataset(image_paths, labels_path, device): 
	'''
	Function to create dataset from multiple sources

	Inputs:
	- image_paths : list of the image paths ( int in this case )
	- labels_path : list of the labels paths (in this case is the bounding boxes)
	- device : string ( 'cuda' )

	Output:
	- all_dataset : ConcatDataset ( The whole mixed dataset )
	'''

	preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

	datasetls = [] 
	for image_path, label_path in zip(image_paths, labels_path):

		dataset = REALLoader(image_path, label_path, device, transformation = preprocess)

		datasetls.append(dataset)

	all_dataset = ConcatDataset(datasetls)


	return all_dataset

def create_test_dataset(image_paths, labels_path, info_paths, device):
	'''
	Function to create dataset from multiple sources

	Inputs:
	- image_paths : list of the image paths
	- labels_path : list of the labels paths (in this case is the bounding boxes)
	output:
	- dataset : Dataset ( Dataset for the test files)
	'''

	preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

	for image_path, label_path, info_path in zip(image_paths, labels_path, info_paths):

		dataset = KITTILoader(image_path, label_path, info_path , device, transformation = preprocess)


	return dataset





def create_loader(dataset):
	'''
	function to create train and val loader

	input :

	- dataset : Dataset (dataset file that was already made)

	Output:

	- train_loader : DatasLoader (training loader)
	- val_loader : Dataloader (validation loader)
	'''

	train_split = int(0.8*len(dataset))
	val_split = len(dataset) - train_split


	train_set, val_set = random_split(dataset,[train_split, val_split])
	train_loader = DataLoader(train_set,batch_size=1, shuffle= True)
	val_loader = DataLoader(val_set,batch_size=1, shuffle= True)
	return train_loader, val_loader



def create_mix_dataset(real_dataset, virtual_dataset):

	'''
	Create an additive dataset to check whether having more virtual data helps in training the object detection

	Input: 
	- real_dataset : Dataset (real images dataset)
	- virtual_dataset : Dataset (virtual dataset)

	Output:
	- mixed dataset : Dataset (mix between real and virtual)
	'''


	mixed_dataset = [real_dataset, virtual_dataset]

	mixed_dataset = ConcatDataset(mixed_dataset)


	return mixed_dataset

def create_mix_50_50_dataset(real_dataset, virtual_dataset):
	'''
	Create a 50 50 mix
	'''
	real_half = int(0.5*len(real_dataset))
	real__ = len(real_dataset) - real_half

	virtual_half = int(0.5*len(virtual_dataset))
	virtual__ = len(virtual_dataset) - virtual_half


	half1, _ = random_split(real_dataset,[real_half, real__])
	half2, _ = random_split(virtual_dataset, [virtual_half, virtual__])

	half_mixed = ConcatDataset([half1, half2])

	return half_mixed


def create_mix_ratio_dataset(ratio,real_dataset, virtual_dataset):
	'''
	Create a ratio based mixed; example ratio = 0.75 would mean that the data consists of 75 % real dataset and 25 % virtual dataset
	

	Inputs:
	- ratio : float ( ratio of the real dataset used )
	- real_dataset : Dataset ( Real dataset )
	- virtual_dataset : Dataset ( Virtual Dataset )

	Output:
	- ratio_mixed : ConcatDataset ( The combined dataset )
	'''
	real_half = int(ratio*len(real_dataset))
	real__ = len(real_dataset) - real_half

	virtual_half = int((1-ratio)*len(virtual_dataset))
	virtual__ = len(virtual_dataset) - virtual_half


	half1, _ = random_split(real_dataset,[real_half, real__])
	half2, _ = random_split(virtual_dataset, [virtual_half, virtual__])

	ratio_mixed = ConcatDataset([half1, half2])

	return ratio_mixed


def create_ratio_virtual_dataset(ratio,real_dataset, virtual_dataset):
	'''
	Create a mix of 100% virtual with a mix of ratio of the real dataset

	Inputs:
	- ratio : float ( ratio of the real dataset used )
	- real_dataset : Dataset ( Real dataset )
	- virtual_dataset : Dataset ( Virtual Dataset )

	Output:
	- ratio_mixed : ConcatDataset ( The combined dataset )
	'''

	real_half = int(ratio*len(real_dataset))
	real__ = len(real_dataset) - real_half

	half1, _ = random_split(real_dataset,[real_half, real__])

	ratio_mixed = ConcatDataset([half1, virtual_dataset])

	return ratio_mixed






def run():
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


	'''
	100 % Virtual Dataset
	'''

	scene_lists = ['Scene01','Scene02'] # Change the umbrella folder name. 6586 rtaining data
	test_scene = 'Test01' # 150 images for the test set


	# here change the sub folder name

	types_list = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-left', 'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']


	test_type = 'test-frames'
	# types_list =['15-deg-left']
	paths = []


	for scene in scene_lists:
		for types in types_list:
			path = os.path.join(scene, types)

			paths.append(path)

	info_paths, bbox_paths = get_labels_bbox(paths)
	image_paths = get_image(paths)

	test_path = os.path.join(test_scene, test_type)

	test_images_paths = get_image([test_path])

	test_info_paths, test_bbox_paths = get_labels_bbox([test_path])


	all_dataset = create_dataset(image_paths, bbox_paths, info_paths, device)

	test_dataset = create_test_dataset(test_images_paths, test_bbox_paths, test_info_paths, device)

	print(f'Virtual Training and Validation set count : {len(all_dataset)}')


	trainloader, valloader = create_loader(all_dataset)

	torch.cuda.empty_cache()

	'''
	100% Virtual Dataset
	'''

	mode = 1
	custom_training(trainloader,valloader, test_dataset, test_images_paths, mode,device = device) # 100% Virtual KITTI, tested against real dataset

	'''
	100 % Real Dataset
	'''

	torch.cuda.empty_cache()


	real_image_path = os.path.join('images', 'Real01', 'training','image_2')
	real_labels_path = os.path.join('labels', 'Real01', 'training','label_2')

	imagesPath =[real_image_path]
	labelsPath=[real_labels_path]

	real_dataset = create_real_dataset(imagesPath, labelsPath, device) # 7481 images

	print(f'Real Training and Validation set count : {len(real_dataset)}')


	real_trainloader, real_valloader = create_loader(real_dataset)

	mode = 2 

	custom_training(real_trainloader,real_valloader, test_dataset, test_images_paths, mode, device=device)

	'''
	Additive Mix Dataset
	'''

	# Create a mix of real and virtual dataset by adding together the real dataset (7000 images) and the virtual dataset (6800 images)

	mode = 3

	torch.cuda.empty_cache()


	mixed_dataset = create_mix_dataset(real_dataset, all_dataset)

	print(f'Mixed Training and validation set count total : {len(mixed_dataset)}')

	mixed_trainloader, mixed_valloader = create_loader(mixed_dataset)

	custom_training(mixed_trainloader,mixed_valloader, test_dataset, test_images_paths, mode, device=device)

	'''
	50-50 mix Of Dataset
	'''

	mode = 4

	torch.cuda.empty_cache()

	mix_50_50_dataset = create_mix_50_50_dataset(real_dataset, all_dataset)

	print(f'50-50 Mixed Training and validation set count total : {len(mix_50_50_dataset)}')

	mixed50_trainloader, mixed50_valloader = create_loader(mix_50_50_dataset)

	custom_training(mixed50_trainloader,mixed50_valloader, test_dataset, test_images_paths, mode, device=device)


	print(f'Test set count {len(test_dataset)}')


	mode = 5 # 75% real, 25% virtual
	torch.cuda.empty_cache()


	ratio = 0.75
	mix_ratio_dataset = create_mix_ratio_dataset(ratio,real_dataset, all_dataset)

	print(f'{ratio} Mixed Training and validation set count total : {len(mix_ratio_dataset)}')

	mixed_ratio_trainloader, mixed_ratio_valloader = create_loader(mix_ratio_dataset)


	custom_training(mixed_ratio_trainloader, mixed_ratio_valloader, test_dataset, test_images_paths, mode, device=device)

	mode = 6 # 75% virtual, 25% real
	torch.cuda.empty_cache()


	ratio = 0.25
	mix_ratio_dataset = create_mix_ratio_dataset(ratio,real_dataset, all_dataset)

	print(f'{ratio} Mixed Training and validation set count total : {len(mix_ratio_dataset)}')

	mixed_ratio_trainloader, mixed_ratio_valloader = create_loader(mix_ratio_dataset)


	custom_training(mixed_ratio_trainloader, mixed_ratio_valloader, test_dataset, test_images_paths, mode, device=device)

	torch.cuda.empty_cache()

	mode = 7  # 100 % Virtual + 75 % Real

	ratio = 0.75
	v_75real_dataset = create_ratio_virtual_dataset(ratio,real_dataset, all_dataset)


	print(f'{ratio} real + 100% virtual Mixed Training and validation set count total : {len(v_75real_dataset)}')

	mixed_ratio_trainloader, mixed_ratio_valloader = create_loader(v_75real_dataset)

	custom_training(mixed_ratio_trainloader, mixed_ratio_valloader, test_dataset, test_images_paths, mode, device=device)

	torch.cuda.empty_cache()
	mode = 8  # 100 % Virtual + 50 % Real

	ratio = 0.5

	v_50real_dataset = create_ratio_virtual_dataset(ratio,real_dataset, all_dataset)

	print(f'{ratio} real + 100% virtual Mixed Training and validation set count total : {len(v_50real_dataset)}')

	mixed_ratio_trainloader, mixed_ratio_valloader = create_loader(v_50real_dataset)

	custom_training(mixed_ratio_trainloader, mixed_ratio_valloader, test_dataset, test_images_paths, mode, device=device)


	torch.cuda.empty_cache()
	mode = 9  # 100 % Virtual + 25 % Real

	ratio = 0.25

	v_25real_dataset = create_ratio_virtual_dataset(ratio,real_dataset, all_dataset)

	print(f'{ratio} real + 100% virtual Mixed Training and validation set count total : {len(v_25real_dataset)}')

	mixed_ratio_trainloader, mixed_ratio_valloader = create_loader(v_25real_dataset)

	custom_training(mixed_ratio_trainloader, mixed_ratio_valloader, test_dataset, test_images_paths, mode, device=device)

	print('Finished')





if __name__ == '__main__':
	run()


