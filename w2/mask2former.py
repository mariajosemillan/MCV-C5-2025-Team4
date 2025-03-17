import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation,Mask2FormerImageProcessor
from tqdm import tqdm
from pycocotools import mask as coco_mask
import json
import cv2
import argparse

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device available:", device)

# processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
# model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance").to(device)
device = "cuda"

model = Mask2FormerForUniversalSegmentation.from_pretrained('/ghome/c5mcv04/MCV-C5-2025-Team4/w2/output_finetune/checkpoint-3140', device_map=device)
processor = Mask2FormerImageProcessor.from_pretrained('/ghome/c5mcv04/MCV-C5-2025-Team4/w2/output_finetune/checkpoint-3140')

def visualize_segmentation(image, segmentation, segments_info, alpha=0.5):
    """
    Overlays the segmentation mask, bounding boxes, and labels on the original image.
    Only objects with class IDs 0 (person) and 2 (car) are visualized.
    """
    image_np = np.array(image.convert("RGB"))
    mask_overlay = np.zeros_like(image_np, dtype=np.uint8)

    np.random.seed(42)  
    colors = {
        segment["id"]: np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        for segment in segments_info
    }

    for segment in segments_info:
        segment_id = segment["id"]
        label_id = segment["label_id"]
        label = model.config.id2label.get(label_id, "Unknown")  # Get class name
        score = segment["score"]

        if label_id not in {0, 1,2}:
            continue

        mask = segmentation == segment_id
        mask_overlay[mask] = colors[segment_id]

        rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))

        bbox = coco_mask.toBbox(rle).tolist() 

        x_min, y_min, width, height = map(int, bbox)
        x_max, y_max = x_min + width, y_min + height
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), colors[segment_id].tolist(), 2)

        label_text = f"{label} ({score:.2f})"
        cv2.putText(image_np, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[segment_id].tolist(), 2)

    mask_boolean = np.any(mask_overlay > 0, axis=-1)
    blended = image_np.copy()
    blended[mask_boolean] = (image_np[mask_boolean] * (1 - alpha) + mask_overlay[mask_boolean] * alpha).astype(np.uint8)

    return blended

def process_single_image(image_path, output_dir):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.size[::-1]], threshold=0.7
    )[0]

    filtered_segments = [segment for segment in results["segments_info"] if segment["label_id"] in {0,1, 2}]

    image_with_masks = visualize_segmentation(image, results['segmentation'].numpy(), filtered_segments)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    Image.fromarray(image_with_masks).save(output_path)
    print(f"Segmented image saved to {output_path}")

def process_images(input_dir, output_dir):
    global image_id_counter, annotation_id_counter
    all_annotations = []

    coco_annotations_file = os.path.join(output_dir, "coco_annotations.json")

    for filename in tqdm(os.listdir(input_dir), desc="Processing Images", unit="image"):
        image_path = os.path.join(input_dir, filename)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs, target_sizes=[image.size[::-1]], threshold=0.7
            )[0]

            filtered_segments = [segment for segment in results["segments_info"] if segment["label_id"] in {0,1, 2}]

            image_with_masks = visualize_segmentation(image, results['segmentation'].numpy(), filtered_segments)

            result_image_path = os.path.join(output_dir, filename)
            Image.fromarray(image_with_masks).save(result_image_path)

            image_info = {
                "id": image_id_counter,
                "file_name": filename,
                "width": image.width,
                "height": image.height
            }
            coco_annotations["images"].append(image_info)

            for segment in filtered_segments:
                mask = results['segmentation'].numpy() == segment["id"]
                rle = coco_mask.encode(np.asfortranarray(mask))
                rle_json = {
                    "counts": rle["counts"].decode("utf-8"),
                    "size": rle["size"]
                }
                annotation = {
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": segment["label_id"],
                    "segmentation": rle_json,
                    "bbox": list(map(int, coco_mask.toBbox(rle))),
                    "area": int(coco_mask.area(rle)),
                    "iscrowd": 0,
                    "score": segment.get("score", 1.0)
                }
                all_annotations.append(annotation)
                annotation_id_counter += 1

            image_id_counter += 1

    coco_annotations["annotations"] = all_annotations
    with open(coco_annotations_file, "w") as json_file:
        json.dump(coco_annotations, json_file, indent=4)

    print(f"COCO annotations saved to {coco_annotations_file}")

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Process images using Mask2Former.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory.")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--single", action="store_true", help="Process a single image instead of a directory.")
    args = parser.parse_args()

   
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "car", "supercategory": "vehicle"}
        ]
    }
    image_id_counter = 1
    annotation_id_counter = 1

    if args.single:
        process_single_image(args.input, args.output)
    else:
        process_images(args.input, args.output)