# cv-project - image segmentation
Steps required to set up data for running the code

## Datasets
The datasets can be found from
Real Data: http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015
Virtual Data: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

Exact download links:
Real Data: http://www.cvlibs.net/download.php?file=data_semantics.zip
Virtual Data:
Original Data: http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar
Segmentation Data: http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_classSegmentation.tar

### Required steps for setting up the code to run

The 2 commands below will extract the virtual dataset from the downloaded tar file.
change directory to data/VKitti_RGB and run command
tar xvf vkitti_2.0.3_rgb.tar

change directory to data/VKitti_classSeg and run command
tar xvf vkitti_2.0.3_classSegmentation.tar

For the real dataset, folder can be extracted normally (File is much smaller for real)

### Run rename_files with the following folder structure in data
As a data folder is created under git ignore, create a new folder to store the dataset
The folder structure for data will be as described

data
├──data_semantics
|  ├──<Extract Real data here>
├──VKitti_classSeg
|  ├──<Extract vkitti_2.0.3_classSegmentation.tar here with command>
├──VKitti_RGB
|  ├──<Extract vkitti_2.0.3_rgb.tar here with command>

## Renaming class segmented files in virtual datasets
The path on line 4 should be updated to match the folder containing the class segmented image