# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os


path = r'C:\Users\teezh\Documents\GitHub\kitti-adapt\data\vKitti_RGB\Scene01\15-deg-left\frames\rgb\Camera_1'
files = os.listdir(path)
i = 1
for file in files:
    filename, file_extension = os.path.splitext(file)
    os.rename(os.path.join(path, file), os.path.join(path, filename.replace('rgb','classgt') + '.png'))