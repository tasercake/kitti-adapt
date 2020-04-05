'''
CV Virtual KITTI object detection

Ivan Christian
'''

from loader.dataloader import KITTILoader
import cv2
import numpy as np
import math


import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist


from torch.utils.data import DataLoader
import os
import torch
from torchvision import transforms
import torchvision.models as models

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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

def create_loader(image_path, target_path,labels_path, info_path, device):
	'''
	Input : 
	- image_path : String ( path to the images )
	- segmentation_path : ( path to masks )
	Output: 
	- dataloader : DataLoader () # To be Updated with train, val, test
	'''

	# preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
	transform = transforms.Compose([transforms.ToTensor()])
	dataset = KITTILoader(image_path, target_path,labels_path, info_path,device, transformation = transform)
	# print(dataset[100])
	dataloader = DataLoader(dataset, batch_size=1, shuffle= True)
	return dataloader


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

	correct = 0

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

		optimizer.zero_grad()
		losses.backward()
		optimizer.step()

		train_loss = loss_value
	return train_loss



def validate ( model , device , val_loader , loss_func):
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

	model.eval().to(device)
	val_loss = 0
	accuracy = 0

	print('Start Validation')

	with torch.no_grad():
		for _, batch in enumerate(val_loader):

			data = batch['images'].to(device)

			targets = batch['target']


			targets = move_to(targets, device)
			loss_dict = model(data, targets) #outputs losses instead of predicition




		print('Validation: ')
		print(loss_dict)



	# 		data = batch['image'].to(device)
	# 		target = batch['label'].long().to(device)

	# 		output = model(data)
	# 		batch_loss = loss_func(output, target).item()#CrossEntropyLoss
	# 		val_loss += batch_loss

	# 		pred = output.argmax(dim=1, keepdim=True)
	# 		accuracy += pred.eq(target.view_as(pred)).sum().item()
	# val_loss /= len(val_loader)

	# accuracy /= len(val_loader.dataset)

	# print(f'Validation set: Average loss: {val_loss}, Accuracy: {accuracy}')
	# return val_loss, accuracy


def custom_training(train_loader, val_loader,device, epochs=5):
	
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

	num_classes = 92

	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	loss_func = nn.CrossEntropyLoss().to(device)


	train_losses_vis = []
	val_losses_vis = []
	val_accuracy_list = []

	print('Starting Training')

	from pprint import pprint

	for epoch in range(1, epochs + 1):
		train_loss = train(model, device, train_loader, optimizer, epoch, loss_func)
		print(train_loss)
		validate(model, device, val_loader, loss_func)
		





def run():
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	path = os.path.join('Scene01', '15-deg-left')
	val_path = os.path.join('Scene01', '15-deg-right')

	labels_path = os.path.join('labels', path)
	segmentation = os.path.join('segmentation', path)

	val_labels_path = os.path.join('labels', val_path)
	val_segmentation = os.path.join('segmentation', val_path)

	img_path = os.path.join('images', path, 'frames', 'rgb', 'Camera_0')

	val_img_path = os.path.join('images', val_path, 'frames', 'rgb', 'Camera_0')

	info_path = os.path.join(labels_path, 'info.txt')
	labels_path = os.path.join(labels_path, 'bbox.txt')

	val_info_path = os.path.join(val_labels_path, 'info.txt')
	val_labels_path = os.path.join(val_labels_path, 'bbox.txt')


	segmentation_path = os.path.join(segmentation, 'frames', 'classSegmentation', 'Camera_0')
	val_segmentation_path = os.path.join(val_segmentation, 'frames', 'classSegmentation', 'Camera_0')


	trainloader = create_loader(img_path, segmentation_path, labels_path, info_path,device)

	valloader = create_loader(val_img_path, val_segmentation_path, val_labels_path, val_info_path, device)

	custom_training(trainloader,valloader, device)


if __name__ == '__main__':
	run()


