from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import os
import cv2
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_predictions(ground_truth_json, prediction_file):
    # Load ground truth and predictions using COCO API
    coco_gt = COCO(ground_truth_json)
    coco_pred = coco_gt.loadRes(prediction_file)
    
    # Evaluate the predictions using COCOeval
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Red
    (0, 0, 255),    # Blue
    (0, 255, 255),  # Cyan
]

def process_images_in_directory(model, processor, device, input_dir, output_dir, subdir, classes_to_detect=[1, 3], threshold=0.5):
    """
    Processes all images in a given directory, performs inference using the model,
    and saves the resulting images with predictions to the output directory.
    """
    input_folder = os.path.join(input_dir, subdir)
    output_folder = os.path.join(output_dir, subdir)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".jpg", ".png", ".jpeg"))]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Processing {len(image_files)} images in {subdir}...")
    
    coco_pred = []
    for image_file in image_files:
        # Load the image
        img = Image.open(image_file).convert("RGB")
        
        # Preprocess the image for DETR
        inputs = processor(images=img, return_tensors="pt").to(device)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-processing (without threshold)
        target_sizes = torch.tensor([img.size[::-1]]).to(device)  # Height x Width
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        
        # Apply thresholding and select classes of interest
        selected_boxes = []
        selected_labels = []
        selected_scores = []
        
        # Now we filter results using the post-processed results
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score.item() >= threshold and label.item() in classes_to_detect:  
                selected_boxes.append(box)
                selected_labels.append(label)
                selected_scores.append(score)
        
        img_cv = np.array(img)
        
        for box, label, score in zip(selected_boxes, selected_labels, selected_scores):
            x0, y0, x1, y1 = box.tolist()
            color = COLORS[label.item() % len(COLORS)] 
            img_cv = cv2.rectangle(img_cv, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
            
            # Display label and confidence score
            label_text = f"{model.config.id2label[label.item()]}: {score:.2f}"
            font_scale = 0.6  # Smaller font size
            thickness = 1     # Thinner font
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            img_cv = cv2.rectangle(img_cv, (int(x0), int(y0) - text_height - 5), (int(x0) + text_width, int(y0)), color, -1)
            img_cv = cv2.putText(img_cv, label_text, (int(x0), int(y0) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            coco_pred.append({
                "image_id": os.path.splitext(os.path.basename(image_file))[0],  
                "category_id": label.item(),
                "bbox": [x0, y0, x1 - x0, y1 - y0],  # COCO format: [x_min, y_min, width, height]
                "score": score.item()
            })
        
        # Save the resulting image with predictions
        output_path = os.path.join(output_folder, os.path.basename(image_file))
        cv2.imwrite(output_path, cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)) 
        print(f"Processed image saved to: {output_path}")
    
    # Create a COCO formatted prediction file
    coco_pred_file = os.path.join(output_dir, "predictions.json")
    with open(coco_pred_file, "w") as f:
        json.dump(coco_pred, f)
    print(f"Predictions saved to: {coco_pred_file}")

def process_all_images(input_dir, output_dir, model, processor, device):
    """Processes all images in the specified directories."""
    subdirs = ["train", "val"]  
    
    for subdir in subdirs:
        process_images_in_directory(model, processor, device, input_dir, output_dir, subdir)
    
    print(f"Inference completed. Results saved in folder: '{output_dir}'.")

if __name__ == "__main__":
    INPUT_DIR = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo/images"  
    OUTPUT_DIR = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/output_DETR"   
    ground_truth_file = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/val_coco_detr.json"
    
    # Process images and generate predictions
    process_all_images(INPUT_DIR, OUTPUT_DIR, model, processor, device)
   
    # Evaluate predictions
    prediction_file = os.path.join(OUTPUT_DIR, "predictions.json")
    evaluate_predictions(ground_truth_file, prediction_file)