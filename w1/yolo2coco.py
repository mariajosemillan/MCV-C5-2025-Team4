import os
import json
from glob import glob
import cv2

dataset_path = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo"
image_dirs = {"train": os.path.join(dataset_path, "images/train"), "val": os.path.join(dataset_path, "images/val")}
label_dirs = {"train": os.path.join(dataset_path, "labels/train"), "val": os.path.join(dataset_path, "labels/val")}
categories = [0,2] 

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

def get_annotation_info(label_path, image_id, annotation_id):
    annotations = []
    with open(label_path, "r") as file:
        lines = file.readlines()
        if not lines:
            print(f"Warning: No annotations found in {label_path}")
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Error in format of {label_path}: {line}")
                continue
            category_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            img_width = 1242
            img_height = 375
            
            xmin = int((x_center - w / 2) * img_width)
            ymin = int((y_center - h / 2) * img_height)
            width = int(w * img_width)
            height = int(h * img_height)
            
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, width, height],
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
        
        if not os.path.exists(image_path):
            print(f"Warning: Could not find image {image_path} for label {label_path}")
            continue

        image_info = get_image_info(image_path, image_id)
        if image_info is None:
            continue
        images.append(image_info)
        anns, annotation_id = get_annotation_info(label_path, image_id, annotation_id)
        annotations.extend(anns)
        image_id += 1
    
    print(f"End of {split}: {len(images)} images y {len(annotations)} annotations processed.")
    
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/{split}_coco.json", "w") as f:
        json.dump(coco_data, f, indent=4)

# Ejecutar conversión
convert_yolo_to_coco("train")
convert_yolo_to_coco("val")
print("Conversion complete")

