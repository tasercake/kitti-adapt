import os
import pandas as pd
import numpy as np

from PIL import Image, ImageDraw
import cv2



path = os.path.join('Scene01', '15-deg-left')

segmentation = os.path.join('segmentation', path)
img_path = os.path.join('images', path, 'frames', 'rgb', 'Camera_0')

mask = os.path.join('labels', path,'colors.txt')


bbox = os.path.join('labels', path, 'bbox.txt')


info = os.path.join('labels', path, 'info.txt')

df = pd.read_csv(bbox, sep = ' ')


infodf = pd.read_csv(info, sep= ' ')

df = df[(df['cameraID']==0)]

df = df[(df['frame']==0)]

df = df.drop(columns = ['number_pixels', 'truncation_ratio', 'occupancy_ratio', 'isMoving'])



infodf['labels'] = infodf['model'] + '_' + infodf['color']
infodf = infodf.drop(columns = ['label','model', 'color'])


print(infodf.shape)



left = np.array(df['left'])
right = np.array(df['right'])
top = np.array(df['top'])
bot = np.array(df['bottom'])
boxes = list(np.stack([left, top,right, bot],axis=1))


labeldf = pd.merge(df, infodf, on='trackID')


print(labeldf)



img = Image.open(os.path.join(img_path, 'rgb_00000.jpg'))


'''
DRAwING THE BOXES ON THE IMAGE
'''

img1 = ImageDraw.Draw(img)
print(len(boxes))
for i in boxes:
	img1.rectangle(list(i), outline ="red")
img.show()