# MCV-C5-2025-Team4

## Session 1: Object Detection and Recognition

### Data
We used the KITTI-MOTS dataset for most training and evaluation tasks. 

For the domain shift task, we will used [Aquarium Dataset from Roboflow](https://public.roboflow.com/object-detection/aquarium), that has 638 images annotated with 7 different classes of aquatic species.

Expected data structure:
```
data
└───KITTI-MOTS
    └───images
    |   └───train
    |   |   └───{sequence_id}_{image_id}.jpg
    |   |       └───...
    |   └───val
    |   |   └───{sequence_id}_{image_id}.jpg
    |   |   └───...
    └───labels
        └───train
        |   └───{sequence_id}_{image_id}.txt
        |   └───...
        └───val
            └───{sequence_id}_{image_id}.txt
            └───...
└───aquarium
    └───train
    |   └───_annotations.coco.json
    |   └───img1.jpg
    |   └───img2.jpg
    |   └───...
    └───val
    |   └───_annotations.coco.json
    |   └───img1.jpg
    |   └───img2.jpg
    |   └───...
    └───test
        └───_annotations.coco.json
        └───img1.jpg
        └───img2.jpg
        └───...
```
