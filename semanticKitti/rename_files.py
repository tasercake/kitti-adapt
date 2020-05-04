# Pythono3 code to rename multiple
import os

root_dir = r'C:\Users\teezh\Documents\GitHub\kitti-adapt\data\vKitti_classSeg'
len_root = len(root_dir)
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        print(os.path.join(subdir[len_root:], file))
        print(file)
        # os.rename(os.path.join(subdir, file), os.path.join(subdir, file.replace('rgb', 'classgt').replace('.jpg','.png')))
        os.rename(os.path.join(subdir, file), os.path.join(subdir, file.replace('classgt', 'rgb').replace('.png','.jpg')))


