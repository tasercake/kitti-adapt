# Pythono3 code to rename multiple
# files in a directory or folder
import glob
# importing os module
import os

root_dir = r'C:\Users\teezh\Documents\GitHub\kitti-adapt\data\vKitti_RGB'
len_root = len(root_dir)
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        print(os.path.join(subdir[len_root:], file))
        print(file)
        # print(os.path.join(subdir, file.replace('rgb', 'classgt').replace('.jpg','.png')))
        # os.rename(os.path.join(subdir, file),
        #           os.path.join(subdir, file.replace('rgb', 'classgt').replace('.jpg','.png')))


