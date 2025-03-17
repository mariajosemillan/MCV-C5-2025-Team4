# MCV-C5-2025-Team4

## Session 2: Object Segmentation

### Setup and Installation
To get started with the project, install the required dependencies for each framework separately:

#### YOLO Environment:
```
pip install -r requirements_yolo.txt
```

#### Detectron2 Environment:
```
pip install -r requirements_detectron.txt
```
Make sure you have the Detectron2, Hugging Face, and Ultralytics frameworks properly set up before proceeding.

⚠️ Note: We use separate virtual environments for YOLO and Detectron2 to avoid version conflicts. Ensure you activate the correct environment before installing dependencies and running models.

### Data
We used the KITTI-MOTS dataset for most training and evaluation tasks. 

For the domain shift task, we used [Aquarium Dataset from Roboflow](https://public.roboflow.com/object-detection/aquarium), that has 638 images annotated with 7 different classes of aquatic species.

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

### Repository Structure

The repository contains the following main scripts:

- detectron2_seg.py: Script to run inference, evaluation or fine-tuning using Detectron2.
- utils.py: Scripts with auxiliary classes and functions for detectron2_seg script.
- yolo_seg.py: Script to run inference, evaluation or fine-tuning using YOLO.
- mask2former.py: Script to run inference, evaluation or fine-tuning using mask2former.
- kitti2yolo.py: Conversion of KITTI-MOTS dataset to YOLO format.
- kitti2coco.py: Conversion of KITTI-MOTS dataset to COCO format.
- [Project_presentation.pdf](Project_presentation.pdf): slides with main tasks and results.
