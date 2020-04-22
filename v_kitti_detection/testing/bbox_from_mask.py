'''
Build Bounding boxes from masks

IC
'''

import os
import numpy as np
import torch
from PIL import Image

import pandas as pd

import cv2




def stack_to_rgb(df):
	r = np.array(df['r'])
	g = np.array(df['g'])
	b = np.array(df['b'])

	mask = np.stack([r,g,b], axis= 1)
	mask = mask[:-1]

	labels = list(df['Category'])[:-1]

	return labels, mask
def bbox(r,c):
	rows = np.any(r, axis=1)
	cols = np.any(c, axis=0)
	rmin, rmax = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))
	cmin, cmax = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))
	return rmin, rmax, cmin, cmax

def process_image(img, mask, label):
	from pprint import pprint

	cv_img = cv2.imread(img)
	
	img = Image.open(img).convert("RGB")



	# cv2.imshow('OG', image)
	# cv2.waitKey(0)

	# exit()

	cv_mask = cv2.imread(mask,1)



	mask = Image.open(mask)

	seg = np.array(mask)

	df = pd.read_csv(label, sep= ' ')
	labels, masks_id = stack_to_rgb(df)
	num_objs = len(labels)

	

	# num_objs =5
	box_label = []
	target_ls = []
	for i in range(num_objs):
		target_ls = show_image_bbox(cv_img, cv_mask, labels[i],masks_id[i]) #returns [{'labels':..., 'boxes': ...}, 
		box_label.extend(target_ls)
		

		# masks = masks_id[i] == seg
		# pos = np.where(masks)

		# # xmin,xmax,ymin,ymax = bbox(pos[0],pos[1])

		# xmin = np.min(pos[1])
		# xmax = np.max(pos[1])
		# ymin = np.min(pos[0])
		# ymax = np.max(pos[0])
		
		# target['boxes'] = [xmin, ymin, xmax, ymax]

		# box_label.append(target)


	xmin, ymin, xmax, ymax = box_label[0]['boxes']

	print(xmin, ymin, xmax, ymax)
	im_arr = np.asarray(img)
	# convert rgb array to opencv's bgr format
	im_arr_bgr = cv2.cvtColor(im_arr, cv2.COLOR_RGB2BGR)

	cv2.rectangle(im_arr_bgr,(xmin, ymin),(xmax, ymax), color=(0, 255, 0), thickness=3)

	im_arr = cv2.cvtColor(im_arr_bgr, cv2.COLOR_BGR2RGB)


	# cv2.imshow('masked',cv_img)
	# cv2.waitKey(0)
	# convert back to Image object
	im = Image.fromarray(im_arr)

	im.show()



	return box_label


def show_image_bbox(ogimg, image, labels, masks_id):


	frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	mask = cv2.inRange(frame, masks_id, masks_id)

	res = cv2.bitwise_and(frame,frame, mask= mask)

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	box_ls = []
	for contour in contours:
		
		# cv2.drawContours(res, contour, -1, (0, 255, 0), 3)

		rect = cv2.boundingRect(contour)

		if rect[2] < 15 or rect[3] < 15: continue

		xmin = int(rect[0])
		ymin = int(rect[1])
		xmax = int(rect[0]+rect[2])
		ymax = int(rect[1]+rect[3])

		cv2.rectangle(ogimg, (int(rect[0]), int(rect[1])),(int(rect[0]+rect[2]), int(rect[1]+rect[3])), (0, 255, 0), 3)


		#Filter out a certain threshold to remove the small/ very small bounding boxes
		
		

		# rect = cv2.minAreaRect(contour)

		# box = cv2.boxPoints(rect)
		# box = np.int0(box) #turn into ints


		# # print('>'*30)
		# # print(box)

		# xmax, ymax = np.amax(box,0)
		# xmin, ymin = np.amin(box,0)


		boxes = [xmin, ymin, xmax, ymax]

		print(boxes)

		target = dict()
		target['labels'] = labels
		target['boxes'] = boxes

		box_ls.append(target)
		# print('>'*30)

		# for i in box:
		# 	cv2.circle(res,(i[0],i[1]), 3, (0,0,255), 2)
			# imgplot = plt.imshow(image)
			# plt.show()
		# print('>'*30)

		# box = box.T
		# print(box) #coordinates --> boxes should be in the form (xmin, ymin)

		# print('>'*30)

		# cv2.drawContours(res, [box], -1, (0, 255, 0), 3)


	cv2.imshow('masked',ogimg)
	cv2.waitKey(0)

	return box_ls



def run():

	image = 'rgb_00000.jpg'
	mask = 'classgt_00000.png'
	text_containing_labels = 'colors.txt'

	target = process_image(image, mask, text_containing_labels)

	print(target)









if __name__ == '__main__':
	run()