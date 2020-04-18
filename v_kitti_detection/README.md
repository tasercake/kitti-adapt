# Object Detection Virtual KITTI Training


## File Structure
```
├───{extracted folder name}
│    ├───images
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
│    │    ├───Scene02
│    │    ├───Scene06
│    │    ├───Scene18
│    │    └───Scene20
│    ├───labels
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
│    │    └───dataloader.py
│    ├───main.py
```
## Sample bbox.txt

frame cameraID trackID left right top bottom number_pixels truncation_ratio occupancy_ratio isMoving
0 0 0 988 1241 159 374 41767 0.4230517 0.7678463 False
0 0 1 927 1102 171 275 7168 0 0.3938462 False
0 0 2 897 984 171 236 2603 0 0.4603006 False
0 0 3 591 666 187 239 2311 0 0.5925641 False
0 0 4 699 732 181 206 452 0 0.5478788 False
0 0 5 847 881 169 195 420 0 0.4751131 False


## Sample info.txt
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

## Loader inputs

- images (processed by the loader)
- info.txt (labels of the bbox[cars, van, truck])
- bbox.txt (bbox coordinates and other features)