# README

##  File Structure
```
├───{extracted folder name}
│    ├───images
│    │    ├───Real01
│    │    │    ├───training
│    │    │    │    └───images_2
│    │    │    └───testing
│    │    │         └───images_2
│    │    ├───Scene01
│    │    │    ├───15-deg-left
│    │    │    │    └───frames
│    │    │    │         └───rgb
│    │    │    │              ├───Camera_0
│    │    │    │              ├───Camera_1
│    │    │    │              └───Unused
│    │    │    ├───15-deg-right
│    │    │    │    └───frames
│    │    │    │         └───rgb
│    │    │    │              ├───Camera_0
│    │    │    │              ├───Camera_1
│    │    │    │              └───Unused
│    │    │    ├───30-deg-left
│    │    │    │    └───frames
│    │    │    │         └───rgb
│    │    │    │              ├───Camera_0
│    │    │    │              ├───Camera_1
│    │    │    │              └───Unused
│    │    │    ├───30-deg-right
│    │    │    │    └───frames
│    │    │    │         └───rgb
│    │    │    │              ├───Camera_0
│    │    │    │              ├───Camera_1
│    │    │    │              └───Unused
│    │    │    ├───clone (same as prev)
│    │    │    ├───fog   
│    │    │    ├───morning
│    │    │    ├───overcast
│    │    │    ├───rain
│    │    │    └───sunset
│    │    ├───Test01
│    │    │    └───test-frames
│    │    │         └───frames
│    │    │              └───rgb
│    │    │                   └───Camera_0
│    │    ├───Scene02
│    │    ├───Scene06
│    │    ├───Scene18
│    │    └───Scene20
│    ├───labels
│    │    ├───Real01
│    │    │    └───training
│    │    │         └───label_2
│    │    ├───Scene01
│    │    │    ├───15-deg-left
│    │    │    │    ├───bbox.txt
│    │    │    │    └───info.txt
│    │    │    └───15-deg-right
│    │    │         ├───bbox.txt
│    │    │         └───info.txt
│    │    ├───Scene02
│    │    ├───Scene06
│    │    ├───Scene18
│    │    └───Scene20
│    ├───loader
│    │    ├───realloader.py
│    │    └───dataloader.py
│    ├───results
│    │    └───sample_bbox
│    │         ├───1
│    │         │   └───(images from Test01)
│    │         ├───2
│    │         │   └───(images from Test01)
│    │         ├───3
│    │         │   └───(images from Test01)
│    │         ├───4
│    │         │   └───(images from Test01)
│    │         ├───5
│    │         │   └───(images from Test01)
│    │         ├───6
│    │         │   └───(images from Test01)
│    │         ├───7
│    │         │   └───(images from Test01)
│    │         ├───8
│    │         │   └───(images from Test01)
│    │         └───9
│    │             └───(images from Test01)
│    ├───object_detect_model1.pt
│    ├───object_detect_model2.pt
│    ├───object_detect_model3.pt
│    ├───object_detect_model4.pt
│    ├───object_detect_model5.pt
│    ├───object_detect_model6.pt
│    ├───object_detect_model7.pt
│    ├───object_detect_model8.pt
│    ├───object_detect_model9.pt
│    └───main.py
```
## INSTRUCTIONS TO RUN
- object_detect_model*.pt are all the weights that we obtained from our training.

### Windows
- Install the the `requirements.txt`

    `pip install requirements.txt`
    
- To evaluate the data with our weights, do not delete the weights and run:

    `python main.py`
    
- To train from scratch to obtain new weights, delete the weights in the folder and run:

    `python main.py`
    
    The results are saved in the `results` folder.

- To add your own image, please copy your image over to Test01/test-frames/frames/rgb/Camera_0. Once your image is copied over, run 

    `python main.py`

    and view you results under the `results` folder. 


### Linux
- Install the the `requirements.txt`

    `pip3 install requirements.txt`
- To start training from scratch, delete the weights in the folder and run:

    `python3 main.py`

- To start testing with our pretrained weights, keep the weights **(.pt files)** in the folder and run:

    `python3 main.py`
    you can view you results under `results` folder.

- To add your own image, please copy your image over to Test01/test-frames/frames/rgb/Camera_0. Once your image is copied over, run 

    `python3 main.py`

    and view you results under the `results` folder. 


## Project Specifications

This project is done using the following devices:

GPU : NVIDIA RTX 2060
CPU : Intel Core i5 - 9400F @ 2.9 GHz
OS : WINDOWS 10 HOME EDITION
Python version : 3.7




