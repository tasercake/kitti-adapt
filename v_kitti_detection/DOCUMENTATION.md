# DOCUMENTATION


##  File Structure
```
├───{extracted folder name}
│    ├───images
│    │    ├───Real01
│    │    │    ├───training
│    │    │    │    ├───images_2
│    │    │    ├───testing
│    │    │    │    ├───images_2
│    │    ├───Scene01
│    │    │    ├───15-deg-left
│    │    │    │    ├───frames
│    │    │    │    │    ├───rgb
│    │    │    │    │    │    ├───Camera_0
│    │    │    │    │    │    ├───Camera_1
│    │    │    │    │    │    ├───Unused
│    │    │    ├───15-deg-right
│    │    │    │    ├───frames
│    │    │    │    │    ├───rgb
│    │    │    │    │    │    ├───Camera_0
│    │    │    │    │    │    ├───Camera_1
│    │    │    │    │    │    ├───Unused
│    │    │    ├───30-deg-left
│    │    │    │    ├───frames
│    │    │    │    │    ├───rgb
│    │    │    │    │    │    ├───Camera_0
│    │    │    │    │    │    ├───Camera_1
│    │    │    │    │    │    ├───Unused
│    │    │    ├───30-deg-right
│    │    │    │    ├───frames
│    │    │    │    │    ├───rgb
│    │    │    │    │    │    ├───Camera_0
│    │    │    │    │    │    ├───Camera_1
│    │    │    │    │    │    ├───Unused
│    │    │    ├───clone (same as prev)
│    │    │    ├───fog   
│    │    │    ├───morning
│    │    │    ├───overcast
│    │    │    ├───rain
│    │    │    ├───sunset
│    │    ├───Test01
│    │    │    ├───training
│    │    │    │    ├───images_2
│    │    │    ├───testing
│    │    │    │    ├───images_2
│    │    ├───Scene02
│    │    ├───Scene06
│    │    ├───Scene18
│    │    └───Scene20
│    ├───labels
│    │    ├───Real01
│    │    │    ├───training
│    │    │    │    ├───label_2
│    │    ├───Scene01
│    │    │    ├───15-deg-left
│    │    │    │    ├───bbox.txt
│    │    │    │    ├───info.txt
│    │    │    ├───15-deg-right
│    │    │    │    ├───bbox.txt
│    │    │    │    ├───info.txt
│    │    ├───Scene02
│    │    ├───Scene06
│    │    ├───Scene18
│    │    └───Scene20
│    ├───loader
│    │    ├───realloader.py
│    │    └───dataloader.py
│    ├───results
│    │    ├───Real01
│    │    │    ├───sample_bbox
│    │    │    │    ├───bbox.txt
│    ├───object_detect_model1.pt
│    ├───object_detect_model2.pt
│    ├───object_detect_model3.pt
│    ├───object_detect_model4.pt
│    ├───object_detect_model5.pt
│    ├───object_detect_model6.pt
│    ├───object_detect_model7.pt
│    ├───object_detect_model8.pt
│    ├───object_detect_model9.pt
│    ├───main.py
```
## INSTRUCTIONS TO RUN

### Windows

To start training from scratch, delete the weights in the folder and run:

`python main.py`

To start testing with our pretrained weights, keep the weights (.pt files) in the folder and run:

`python main.py`


### Linux

To start training from scratch, delete the weights in the folder and run:

`python3 main.py`

To start testing with our pretrained weights, keep the weights (.pt files) in the folder and run:

`python3 main.py`


## VIRTUAL DATASET

### Sample Image

![](https://i.imgur.com/5PAH9FS.jpg)

### Sample bbox.txt
```
frame cameraID trackID left right top bottom number_pixels truncation_ratio occupancy_ratio isMoving
0 0 0 988 1241 159 374 41767 0.4230517 0.7678463 False
0 0 1 927 1102 171 275 7168 0 0.3938462 False
0 0 2 897 984 171 236 2603 0 0.4603006 False
0 0 3 591 666 187 239 2311 0 0.5925641 False
0 0 4 699 732 181 206 452 0 0.5478788 False
0 0 5 847 881 169 195 420 0 0.4751131 False
```

### Sample info.txt
```
trackID label model color
0 Car Sedan4Door Black
1 Car Hatchback Black
2 Car Hybrid Black
3 Car Hatchback Red
4 Car Hatchback Silver
5 Car Hatchback Blue
6 Car Sedan4Door Red
7 Car SUV Black
9 Car Hatchback Black
```
### Loader inputs

- images (processed by the loader)
- info.txt (labels of the bbox [cars, van, truck])
- bbox.txt (bbox coordinates and other features)

## REAL DATASET

### Sample Image

![](https://i.imgur.com/ftnitA4.png)

### sample (image).txt
```
Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56
Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
Cyclist 0.00 3 -1.65 676.60 163.95 688.98 193.93 1.86 0.60 2.02 4.59 1.32 45.84 -1.55
```
### Loader inputs

- images (processed by the loader)
- (image id).txt (bbox coordinates and other features)

## Current state of the experiment

Finetuning a Default Faster RCNN using the following hyperparameters.

```python=
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

num_classes = 4 #Car, Truck, Van and Background

in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
loss_func = nn.BCEWithLogitsLoss().to(device) #Could have been better in 

```

## Training, Validation, Testing

### Model Testing

Testing the model will be done on the same dataset 


### 100 % Virtual KITTI Dataset Training (mode = 1)

Virtual Training and Validation set count : 6586

Mean Training Loss: 
- Epoch 1 Train Loss = `0.0736868984290947`
- Epoch 2 Train Loss = `0.03630958454255241`
- Epoch 3 Train Loss = `0.026153916658399782`
- Epoch 4 Train Loss = `0.021074651686394066`

Mean Validation Accuracy:
- Epoch 1 Validation Accuracy = `0.567833662033081`
- Epoch 2 Validation Accuracy = `0.6165010333061218`
- Epoch 3 Validation Accuracy = `0.737072229385376`
- Epoch 4 Validation Accuracy = `0.7934923768043518`


Mean Test Confidence Score : `0.4702227`

#### Sample Bounding Box Visualisation:

![](https://i.imgur.com/6xw8wLN.png)


### 100 % Real Dataset Training (mode = 2)

Real Training and Validation set count : 7481

Training Loss: 
- Epoch 1 Train Loss = `0.08353609897886614`
- Epoch 2 Train Loss = `0.05013361077763549`
- Epoch 3 Train Loss = `0.04098432157263748`
- Epoch 4 Train Loss = `0.03301855931225445`

Validation Accuracy:
- Epoch 1 Validation Accuracy = `0.4099167287349701`
- Epoch 2 Validation Accuracy = `0.484778493642807`
- Epoch 3 Validation Accuracy = `0.4686709940433502`
- Epoch 4 Validation Accuracy = `0.5394859313964844`

Mean Test Accuracy: `0.38511887`

#### Sample Bounding Box Visualisation:

![](https://i.imgur.com/6fsGc5n.png)


### 100% Real + 100% Virtual Mix Dataset (mode = 3)

Mixed Training and validation set count total : 14067


Training Loss: 
- Epoch 1 Train Loss = `0.0702982111942408`
- Epoch 2 Train Loss = `0.04173062674887977`
- Epoch 3 Train Loss = `0.031732824132320904`
- Epoch 4 Train Loss = `0.024971065365085525`

Validation Accuracy:
- Epoch 1 Validation Accuracy = `0.4740722179412842`
- Epoch 2 Validation Accuracy = `0.527308464050293`
- Epoch 3 Validation Accuracy = `0.6535086035728455`
- Epoch 4 Validation Accuracy = `0.6861435174942017`


Mean Test Accuracy: `0.5486722`

#### Sample Bounding Box Visualisation:


![](https://i.imgur.com/2266CVn.png)

### 50 % Real + 50 % Virtual Mix Dataset (mode = 4)

50-50 Mixed Training and validation set count total : 7033

Training Loss: 
- Epoch 1 Train Loss = `0.09077293264259224`
- Epoch 2 Train Loss = `0.054096570048142535`
- Epoch 3 Train Loss = `0.041036723170948766`
- Epoch 4 Train Loss = `0.03312976428472635`

Validation Accuracy:
- Epoch 1 Validation Accuracy = `0.3858115077018738`
- Epoch 2 Validation Accuracy = `0.5489523410797119`
- Epoch 3 Validation Accuracy = `0.5552843284321308136`
- Epoch 4 Validation Accuracy = `0.6292757391929626`

Mean Test Accuracy: `0.47117433`

#### Sample Bounding Box Visualisation:

![](https://i.imgur.com/z9uMlKK.png)



### 75 % Real + 25 % Virtual Dataset (mode = 5)

0.75 Mixed Training and validation set count total : 7256

Training Loss: 
- Epoch 1 Train Loss = `0.08905999011786617`
- Epoch 2 Train Loss = `0.057487303320285`
- Epoch 3 Train Loss = `0.04437502387031428`
- Epoch 4 Train Loss = `0.036155355913961865`

Validation Accuracy:
- Epoch 1 Validation Accuracy = `0.41694176197052`
- Epoch 2 Validation Accuracy = `0.48023903369903564`
- Epoch 3 Validation Accuracy = `0.5440050363540649`
- Epoch 4 Validation Accuracy = `0.6022129058837891`


Mean Test Accuracy: `0.4913343`

#### Sample Bounding Box Visualisation:

![](https://i.imgur.com/WbZSEE2.png)


### 25 % Real + 75 % Virtual Dataset Training (mode = 6)

0.25 Mixed Training and validation set count total : 6809

Training Loss: 
- Epoch 1 Train Loss = `0.08369456324083034`
- Epoch 2 Train Loss = `0.04800657526663793`
- Epoch 3 Train Loss = `0.03592450226087861`
- Epoch 4 Train Loss = `0.02861895240844557`

Validation Accuracy:
- Epoch 1 Validation Accuracy = `0.4681207239627838`
- Epoch 2 Validation Accuracy = `0.5764805674552917`
- Epoch 3 Validation Accuracy = `0.621878087505994`
- Epoch 4 Validation Accuracy = `0.6917305588722229`


Mean Test Accuracy: `0.49066687`

#### Sample Bounding Box Visualisation:

![](https://i.imgur.com/ci44Pj1.png)

### 75 % Real + 100% Virtual Dataset (mode = 7)

75 % Real + 100% Virtual Mixed Training and validation set count total : 12196

Training Loss: 
- Epoch 1 Train Loss = `0.07579848395798748`
- Epoch 2 Train Loss = `0.043899508051786586`
- Epoch 3 Train Loss = `0.033082823642401296`
- Epoch 4 Train Loss = `0.026088542501726993`

Validation Accuracy:
- Epoch 1 Validation Accuracy = `0.48124945163726807`
- Epoch 2 Validation Accuracy = `0.539775192737593`
- Epoch 3 Validation Accuracy = `0.6866471147537231`
- Epoch 4 Validation Accuracy = `0.6725687980651855`


Mean Test Accuracy: `0.5276988`

#### Sample Bounding Box Visualisation:

![](https://i.imgur.com/lWgqnef.png)

### 50 % Real + 100 % Virtual Dataset (mode = 8)

50 % Real + 100 % Virtual Mixed Training and validation set count total : 10326


Training Loss: 
- Epoch 1 Train Loss = `0.07672068978270537`
- Epoch 2 Train Loss = `0.04571226874124552`
- Epoch 3 Train Loss = `0.03524815633047522`
- Epoch 4 Train Loss = `0.02813912194254839`

Validation Accuracy:
- Epoch 1 Validation Accuracy = `0.4769132733345032`
- Epoch 2 Validation Accuracy = `0.5596204996109009`
- Epoch 3 Validation Accuracy = `0.6303060054779053`
- Epoch 4 Validation Accuracy = `0.7019739747047424`


Mean Test Accuracy: `0.5292934`

#### Sample Bounding Box Visualisation:


### 25 % Real + 100% Virtual Dataset (mode = 9)

25 % Real + 100% Virtual Training and validation set count total : 8456

Training Loss: 
- Epoch 1 Train Loss = `0.07740479824470553`
- Epoch 2 Train Loss = `0.04172045023216756`
- Epoch 3 Train Loss = `0.03162234577172598`
- Epoch 4 Train Loss = `0.025374139274913794`

Validation Accuracy:
- Epoch 1 Validation Accuracy = `0.5430802702903748`
- Epoch 2 Validation Accuracy = `0.5831497311592102`
- Epoch 3 Validation Accuracy = `0.5451768636703491`
- Epoch 4 Validation Accuracy = `0.7485036849975586`


Mean Test Accuracy: `0.5199169`

#### Sample Bounding Box Visualisation:

