from loader.dataloader import KITTILoader
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

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt

from PIL import Image

import glob

def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
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

			losses = sum(loss for loss in loss_dict.values())

			# reduce losses over all GPUs for logging purposes
			loss_dict_reduced = reduce_dict(loss_dict)
			losses_reduced = sum(loss for loss in loss_dict_reduced.values())

			loss_value = losses_reduced.item()

			# print(loss_value)

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

	# with tqdm(total = len(val_loader)) as bar:

	with torch.no_grad():
		for idx , batch in enumerate(val_loader):
			data = batch['images'].to(device)
			targets = batch['target']
			targets = move_to(targets, device)
			prediction = model(data, targets) #outputs prediction instead of losses in eval mode.
			output = prediction[0]['scores']

			mean_score = output.mean()

			print(mean_score)

			val_accuracy += mean_score

				# bar.update()

	mean_val_accuracy  = val_accuracy / len(val_loader)

	print(f'Epoch {epochs} Validation Accuracy = {mean_val_accuracy}')
	return mean_val_accuracy

def image_scores(path, models, model_path, output_path, test_dataset, transformation, device = 'cuda'):
	'''
	Function is used to calculate the image prediction scores

	Input :
	- path : string (root directory for the dataset)
	- models : model (model used in this task)
	- model_path : string ( model directory )
	- test_dataset : Dataset ( essentially the testing files )
	- transformation : Transform ( Augmentation )
	- device : string ('cuda')
	'''
	print(f'Loading model to score images. Scores saved {output_path}') # Testing the model

def custom_training(train_loader, val_loader,device, epochs=3):
	
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

	num_classes = 10

	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	loss_func = nn.BCEWithLogitsLoss().to(device)


	train_losses_vis = []
	val_losses_vis = []
	val_accuracy_list = []

	print('Starting Training')

	for epoch in range(1, epochs):
		train_loss = train(model, device, train_loader, optimizer, epoch, loss_func)
		accuracy = validate(model, device, val_loader, loss_func, epoch)

		if (len(train_losses_vis) > 0) and (train_loss < min(train_losses_vis)):
			torch.save(model.state_dict(), 'object_detect_model.pt')

		train_losses_vis.append(train_loss)

		val_accuracy_list.append(accuracy)

	return train_losses_vis, val_accuracy_list

def create_dataset(image_paths, labels_path, device):
	'''
	Function to create dataset from multiple sources

	Inputs:
	- image_paths : list of the image paths
	- labels_path : list of the labels paths (in this case is the bounding boxes)
	- 
	'''

	preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

	datasetls = [] 
	for image_path, label_path in zip(image_paths, labels_path):

		dataset = KITTILoader(image_path, label_path, device, transformation = preprocess)

		datasetls.append(dataset)

	all_dataset = ConcatDataset(datasetls)


	return all_dataset

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



def run():
	
	device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	imagesPath =["imagesPath"]
	labelsPath=["labelPath"]
	all_dataset = create_dataset(imagesPath, labelsPath, device)

	print(len(all_dataset))

	trainloader, valloader = create_loader(all_dataset)

	#torch.cuda.empty_cache()


	if os.path.exists('object_detect_model.pt'):

		print('Weights exists')
	else:
		print('weights not available. Starting training')
		train_losses, val_accuracy = custom_training(trainloader,valloader, device)

		print(train_losses, val_accuracy)