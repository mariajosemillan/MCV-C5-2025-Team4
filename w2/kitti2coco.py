
import os
import json
from glob import glob
import cv2
from pycocotools import mask

dataset_path = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo_masks"
image_dirs = {"train": os.path.join(dataset_path, "images/train"), "test": os.path.join(dataset_path, "images/test"),"val": os.path.join(dataset_path, "images/val")}
label_dirs = {"train": os.path.join(dataset_path, "labels/train"), "test": os.path.join(dataset_path, "labels/test"),"val": os.path.join(dataset_path, "labels/val")}
categories = [  { "id": 0, "name": "person" },
                { "id": 1, "name": "bicycle" },
                { "id": 2, "name": "car" },
                { "id": 3, "name": "motorcycle" },
                { "id": 4, "name": "airplane" },
                { "id": 5, "name": "bus" },
                { "id": 6, "name": "train" },
                { "id": 7, "name": "truck" },
                { "id": 8, "name": "boat" },
                { "id": 9, "name": "traffic light" },
                { "id": 10, "name": "fire hydrant" },
                { "id": 11, "name": "stop sign" },
                { "id": 12, "name": "parking meter" },
                { "id": 13, "name": "bench" },
                { "id": 14, "name": "bird" },
                { "id": 15, "name": "cat" },
                { "id": 16, "name": "dog" },
                { "id": 17, "name": "horse" },
                { "id": 18, "name": "sheep" },
                { "id": 19, "name": "cow" },
                { "id": 20, "name": "elephant" },
                { "id": 21, "name": "bear" },
                { "id": 22, "name": "zebra" },
                { "id": 23, "name": "giraffe" },
                { "id": 24, "name": "backpack" },
                { "id": 25, "name": "umbrella" },
                { "id": 26, "name": "handbag" },
                { "id": 27, "name": "tie" },
                { "id": 28, "name": "suitcase" },
                { "id": 29, "name": "frisbee" },
                { "id": 30, "name": "skis" },
                { "id": 31, "name": "snowboard" },
                { "id": 32, "name": "sports ball" },
                { "id": 33, "name": "kite" },
                { "id": 34, "name": "baseball bat" },
                { "id": 35, "name": "baseball glove" },
                { "id": 36, "name": "skateboard" },
                { "id": 37, "name": "surfboard" },
                { "id": 38, "name": "tennis racket" },
                { "id": 39, "name": "bottle" },
                { "id": 40, "name": "wine glass" },
                { "id": 41, "name": "cup" },
                { "id": 42, "name": "fork" },
                { "id": 43, "name": "knife" },
                { "id": 44, "name": "spoon" },
                { "id": 45, "name": "bowl" },
                { "id": 46, "name": "banana" },
                { "id": 47, "name": "apple" },
                { "id": 48, "name": "sandwich" },
                { "id": 49, "name": "orange" },
                { "id": 50, "name": "broccoli" },
                { "id": 51, "name": "carrot" },
                { "id": 52, "name": "hot dog" },
                { "id": 53, "name": "pizza" },
                { "id": 54, "name": "donut" },
                { "id": 55, "name": "cake" },
                { "id": 56, "name": "chair" },
                { "id": 57, "name": "couch" },
                { "id": 58, "name": "potted plant" },
                { "id": 59, "name": "bed" },
                { "id": 60, "name": "dining table" },
                { "id": 61, "name": "toilet" },
                { "id": 62, "name": "tv" },
                { "id": 63, "name": "laptop" },
                { "id": 64, "name": "mouse" },
                { "id": 65, "name": "remote" },
                { "id": 66, "name": "keyboard" },
                { "id": 67, "name": "cell phone" },
                { "id": 68, "name": "microwave" },
                { "id": 69, "name": "oven" },
                { "id": 70, "name": "toaster" },
                { "id": 71, "name": "sink" },
                { "id": 72, "name": "refrigerator" },
                { "id": 73, "name": "book" },
                { "id": 74, "name": "clock" },
                { "id": 75, "name": "vase" },
                { "id": 76, "name": "scissors" },
                { "id": 77, "name": "teddy bear" },
                { "id": 78, "name": "hair drier" },
                { "id": 79, "name": "toothbrush" }]

def get_image_info(image_path, image_id):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    height, width, _ = image.shape
    return {
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "width": width,
        "height": height
    }

def get_annotation_info(seq,img, image_id, annotation_id):
    annotations = []
    label_dir="/ghome/c5mcv04/mcv/datasets/C5/KITTI-MOTS/instances_txt"
    label_path=os.path.join(label_dir,seq+'.txt')
    with open(label_path, "r") as file:
        lines = file.readlines()
        if not lines:
            print(f"Warning: No annotations found in {label_path}")
        for line in lines:
            parts = line.strip().split()
            
            if len(parts) < 5 or int(parts[0]) != int(img):
                #print(f"{parts[0]} deberian ser dintintas {img}")
                continue
            class_id = int(parts[2])
            # if class_id == 2:
            #     class_id = 0 # 2 = Car in Yolo
            
            if class_id == 1:
                class_id = 2 # 2 = Car in Yolo
            elif class_id == 2:
                class_id = 0 # 0 = Person in Yolo

            width= int(parts[4])
            height= int(parts[3])
            
            if class_id != 10:
                segmentation = {'size': [height, width], 'counts': parts[5]}
                bbox = mask.toBbox(segmentation).tolist()

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": width * height,
                    "iscrowd": 0
                }
                annotations.append(annotation)
                annotation_id += 1
    return annotations, annotation_id

def convert_yolo_to_coco(split):
    images, annotations = [], []
    image_id, annotation_id = 1, 1
    print(os.path.join(label_dirs[split]))
    for label_path in sorted(glob(os.path.join(label_dirs[split], "*.txt"))):
        image_name = os.path.basename(label_path).replace(".txt", ".png") 
        image_path = os.path.join(image_dirs[split], image_name)
        seq = image_name.split('_')[0]
        img = image_name.split('_')[1].split('.')[0]

        if not os.path.exists(image_path):
            print(f"Warning: Could not find image {image_path} for label {label_path}")
            continue

        image_info = get_image_info(image_path, image_id)
        if image_info is None:
            continue
        images.append(image_info)
        anns, annotation_id = get_annotation_info(seq,img, image_id, annotation_id)
        annotations.extend(anns)
        image_id += 1
    
    print(f"End of {split}: {len(images)} images y {len(annotations)} annotations processed.")
    
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/{split}_coco_all.json", "w") as f:
        json.dump(coco_data, f, indent=4)

# Ejecutar conversión
#convert_yolo_to_coco("test")
convert_yolo_to_coco("test")
#convert_yolo_to_coco("val")
print("Conversion complete")