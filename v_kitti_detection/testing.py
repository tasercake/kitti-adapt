import os
import pandas as pd
import numpy as np

from PIL import Image, ImageDraw
import cv2



path = os.path.join('Scene01', '15-deg-left')

# img_path = os.path.join('images',path,'frames','rgb','Camera_0')
# labels_path = os.path.join('labels', path)

# labels = os.path.join(labels_path, 'bbox.txt')



#  # , ,



# left = np.array(df['left'])
# right = np.array(df['right'])
# top = np.array(df['top'])
# bot = np.array(df['bottom'])
# boxes = np.stack([left,right, top, bot],axis=1)

# print(list(boxes[0]))


# img = Image.open(os.path.join(img_path, 'rgb_00000.jpg'))


# img1 = ImageDraw.Draw(img)

# img1.rectangle(list(boxes[5]), outline ="red")

# img.show()



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



# print(df)




infodf['labels'] = infodf['model'] + '_' + infodf['color']
infodf = infodf.drop(columns = ['label','model', 'color'])


# infodf = infodf.to_dict() # dictionary of 

print(infodf.shape)



left = np.array(df['left'])
right = np.array(df['right'])
top = np.array(df['top'])
bot = np.array(df['bottom'])
boxes = list(np.stack([left, top,right, bot],axis=1))


labeldf = pd.merge(df, infodf, on='trackID')


print(labeldf)


# result = [df, infodf]

# result = pd.concat(result)

# print(result)
# print(boxes)
# boxes = list(boxes)



'''
Segementation using only the masks

'''


# segmentation_path = os.path.join(segmentation, 'frames', 'classSegmentation', 'Camera_0')

img = Image.open(os.path.join(img_path, 'rgb_00000.jpg'))

# 

df = pd.read_csv(mask, sep= ' ')
r = np.array(df['r'])
g = np.array(df['g'])
b = np.array(df['b'])


mask = np.stack([r,g,b], axis= 1)

mask = mask[:-1]



labels = list(df['Category'])[:-1]
print(labels)
target = dict()

for i in range(len(labels)):
	target[labels[i]] = mask[i]

print(target)

img1 = ImageDraw.Draw(img)
print(len(boxes))
for i in boxes:
	img1.rectangle(list(i), outline ="red")
img.show()



# # import cv2

# img = cv2.imread(os.path.join(img_path, 'rgb_00000.jpg'))


# # res = cv2.bitwise_and(img,img, mask=mask)
# # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

# # print(res)

# # image = Image.fromarray(res)

# # image.show()


# seg = Image.open(os.path.join(segmentation_path, 'classgt_00000.png'))

# # pil_image = Image.composite(seg,mask[0],seg)

# # pil_image.show()

# seg = np.array(seg)


# num_objs = len(labels)
# boxes = []
# for i in range(num_objs-1,num_objs):
# 	print(mask[i])
# 	masks = mask[i] == seg
# 	pos = np.where(masks)


# 	# print(pos)
# 	xmin = np.min(pos[1])
# 	xmax = np.max(pos[1])
# 	ymin = np.min(pos[0])
# 	ymax = np.max(pos[0])
# 	boxes.append([xmin, ymin, xmax, ymax])

# 	# print(boxes)







# labels_path = os.path.join('labels', path)

# info_path = os.path.join(labels_path, 'info.txt')

# labels_path = os.path.join(labels_path, 'bbox.txt')


# info_df = pd.read_csv(info_path, sep=" ")

# df = pd.read_csv(labels_path, sep=" ")

# df = df[(df['cameraID']==0)]



# df_bot = np.array(df['bottom'])
# df_top = np.array(df['top'])
# df_left = np.array(df['left'])
# df_right = np.array(df['right'])




# print(info_df.drop(columns = 'label'))