'''
Utils for visualisation
'''

import os

from PIL import Image, ImageDraw 


def save_image(image,directory, filename):
	if not os.path.exists(directory):
		os.makedirs(directory)

	img_file = os.path.join(directory, filename)

	image.save(img_file)

	print(f'Saved to {img_file}')