"""
results = model.predict(
                        source="input.jpg",  # Image, video, or folder with images
                        conf=0.5,  # Confidence threshold (default is 0.25)
                        iou=0.5,  # IoU threshold for Non-Maximum Suppression (NMS)
                        save=True,  # Save results
                        save_txt=True,  # Save results in TXT format
                        save_crop=True,  # Save cropped objects
                        show=True,  # Display the image with predictions
                        line_width=2,  # Line thickness for bounding boxes
                        box=True,  # Draw boxes around detected objects
                        conf_label=True,  # Show confidence score in prediction
                        device="cuda",  # Specify GPU or CPU ("cpu" or "cuda")
                        classes=[0, 2],  # Filter by specific classes (e.g., 0: person, 2: car)
                        imgsz=640,  # Input image size (default is 640)
                        augment=True,  # Use data augmentation during inference
                        half=True  # Use 16-bit precision (float16) for faster GPU processing
                    )
"""
"""
model.val(
            data="dataset.yaml",  # Path to the dataset YAML file
            split="val",  # Dataset split ('train', 'val', or 'test')
            batch=16,  # Batch size
            imgsz=640,  # Image size
            conf=0.001,  # Minimum confidence threshold for detection
            iou=0.6,  # IoU threshold for Non-Maximum Suppression (NMS)
            device="cuda",  # Specify device ('cuda' for GPU, 'cpu' for CPU)
            workers=8,  # Number of worker processes for data loading
            save_json=True,  # Save results in COCO JSON format
            save_hybrid=False,  # Save images with combined predictions and labels
            save_txt=False,  # Save predictions in YOLO (txt) format
            save_conf=False,  # Save confidence scores in text files
            save_crop=False,  # Save detected objects as cropped images
            save=True,  # Save images with annotations
            half=False,  # Use FP16 (half precision) to improve performance on compatible GPUs
            augment=False,  # Use augmentation during validation
            rect=False,  # Use proportional resizing in validation
            vid_stride=1,  # Stride for videos (useful if validating on video sequences)
            plots=True,  # Generate and save precision, recall, and mAP plots
            name="exp",  # Experiment name for results folder
            exist_ok=False,  # Overwrite previous results
            verbose=True,  # Show additional details in the output
            project="runs/val",  # Folder where results will be saved
            classes=[0, 2],  # Filter detections only for specific classes (0=person, 2=car in COCO)
        )
"""
import os
import sys
import shutil
import argparse
from ultralytics import YOLO

def move_files_from_batch_to_main(subdir):
    """Move all files from the current batch folder to the main folder."""
    main_folder = os.path.join(OUTPUT_DIR, "images")
    
    # Check if there are any new subfolders (train2, train3, etc.)
    subfolders = [folder for folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder))]
    print(f"SUBFOLDERS: {subfolders}")
    for subfolder in subfolders:
        if subfolder != 'train' and subfolder != 'val' and subfolder != 'test':
            subfolder_path = os.path.join(main_folder, subfolder)
            print(f"Moving files from {subfolder_path} to {os.path.join(main_folder, subdir)}...")
            
            # Move all files from the subfolder to the main folder
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                if os.path.isfile(file_path):
                    shutil.move(file_path, os.path.join(os.path.join(main_folder, subdir), filename))
            
            # After moving, delete the subfolder
            os.rmdir(subfolder_path)
            print(f"Deleted subfolder {subfolder_path}")

def process_images_in_batches(model, image_list, batch_size, subdir):
    """Process images in batches to avoid too many open files error."""
    total_images = len(image_list)
    for i in range(0, total_images, batch_size):
        batch = image_list[i:i + batch_size]
        print(f"Processing images {i+1} - {i+len(batch)} of {total_images} in {subdir}...")

        # Run inference on the current batch of images
        model.predict(
            source=batch,  # Process only this batch
            save=True,  # Save images with predictions
            project=OUTPUT_DIR,  # Maintain same directory structure
            name=f"images/{subdir}",  # Always save in the same folder (train/ or val/)
            conf=0.5,  # Confidence threshold
            iou=0.5,  # IoU threshold for non-maximum suppression (NMS)
            classes=[0, 1],  # 0: person, 2: car
            device="cuda"  # Use GPU if available
        )

        # Move files after processing this batch
        move_files_from_batch_to_main(subdir)

def yolo_run_inference(MODEL_PATH,DATASET_DIR,OUTPUT_DIR,BATCH_SIZE):
    """Run inference with YOLO."""
    try:
        model = YOLO(MODEL_PATH)

        # Folders to process (train and val)
        #subdirs = ["train", "val"]
        subdirs = ["test"]

        # Process each folder (train and val)
        for subdir in subdirs:
            input_folder = os.path.join(DATASET_DIR, "images", subdir)
            output_folder = os.path.join(OUTPUT_DIR, "images", subdir)

            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Get all image files in the folder
            image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

            if not image_files:
                print(f"No images found in {input_folder}")
                continue

            # Process images in batches
            process_images_in_batches(model, image_files, BATCH_SIZE, subdir)

        print(f"Inference completed. Check the '{OUTPUT_DIR}' folder")

    except Exception as e:
        print(f"Error executing {MODEL_NAME}: {e}")
        sys.exit(1)

def yolo_run_eval(MODEL_PATH,DATASET_CONFIG):
    """Run evaluation (validation) with YOLO."""
    try:
        # Load YOLOv8 model
        model = YOLO(MODEL_PATH)

        # Run validation
        results = model.val(
            data=DATASET_CONFIG,  # Path to dataset.yaml
            imgsz=640,  # Image size for evaluation
            batch=16,  # Batch size
            conf=0.5,  # Confidence threshold
            iou=0.5,  # IoU threshold
            plots=True,
            device="cuda"  # Use GPU if available
        )

        print("mAP50-95:",results.seg.map)  # map50-95(M)
        print("mAP50:",results.seg.map50)  # map50(M)
        print("mAP75:",results.seg.map75)  # map75(M)
        print("Mean Average Precision for masks:", results.seg.map)# a list contains map50-95(M) of each category

    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)

def yolo_run_train(MODEL_PATH,DATASET_CONFIG_FINETUNE):
    # # model = YOLO(MODEL_PATH)
    # # train_results = model.train(
    # #     data=DATASET_CONFIG_FINETUNE,  # path to dataset YAML
    # #     epochs=2,  # number of training epochs
    # #     imgsz=640,  # training image size
    # # )
    yolov11_configs = [
        {
            "name": "freeze_backbone",
            "epochs": 20,
            "batch": 8,
            "lr0": 0.001,
            "freeze": 10,  
            "patience": 50
        },
        {
            "name": "freeze_backbone_neck",
            "epochs": 20,
            "batch": 8,
            "lr0": 0.001,
            "freeze": ["backbone", "neck"] ,  
            "patience": 50
        },
        {
            "name": "freeze_all__cls1_dlf2_yolo",
            "epochs": 30,
            "batch": 8,
            "lr0": 0.0005,
            "freeze": 10,  
            "cls": 1.0,
            "dfl": 2.0,
            "crop_fraction":0.3,
        },
        {
            "name": "freeze_all__several_params",
            "epochs": 20,
            "batch": 8,
            "lr0": 0.0005,
            "freeze": 10,  
            "cls": 1.0,
            "dfl": 2.0,
            "hsv_s": 0.3,
            "translate":0.2,
            "scale":0.4,
            "crop_fraction": 0.3,
        },
     ]

    for config in yolov11_configs:
        cache= "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo_masks_finetune/labels"
        file_val=os.path.join(cache, "val.cache")
        file_train=os.path.join(cache, "train.cache")
        if os.path.exists(file_val):  
            os.remove(file_val)  
            print(f"Delete: {file_val}")
        if os.path.exists(file_train): 
            os.remove(file_train)  
            print(f"Delete: {file_train}")
        print(f"Training with configuration: {config['name']}")
        
        model = YOLO(MODEL_PATH)  
        
        train_results = model.train(
            data=DATASET_CONFIG_FINETUNE,
            epochs=config["epochs"],
            batch=config["batch"],
            lr0=config["lr0"],
            optimizer="AdamW",
            #momentum=config["momentum"],
            #weight_decay=config["weight_decay"],
            freeze=config.get("freeze", None),
            warmup_epochs=config.get("warmup_epochs", 3),
            multi_scale=config.get("multi_scale", False),
            close_mosaic=config.get("close_mosaic", 0),
            rect=config.get("rect", False),
            fraction=config.get("fraction", 1.0),
            cos_lr=config.get("cos_lr", False),
            patience=config.get("patience", 100),
            label_smoothing=config.get("label_smoothing", 0.0),
            imgsz=config.get("imgsz", 640),
            #box=config.get("box", 7.5),
            cls=config.get("cls", 0.5),
            dfl=config.get("dfl", 1.5),
            hsv_h=config.get("hsv_h", 0.0), 
            hsv_s= config.get("hsv_s", 0.0),
            hsv_v= config.get("hsv_v", 0.0),  
            #fliplr= config.get("fliplr", None), 
            translate= config.get("translate", 0.0), 
            scale= config.get("scale", 0.0), 
            crop_fraction= config.get("crop_fraction", 0.0),
            name=config["name"]
        )

        results_dir = os.path.join("runs", "segment", config["name"])
        os.makedirs(results_dir, exist_ok=True)  

        results_file = os.path.join(results_dir, "training_results.txt")

        with open(results_file, "w") as f:
            f.write(f"Training with: {config['name']} is ready.\n\n")
            f.write(f"Final Results: {train_results.seg}\n")
            f.write(f"mAP@50: {train_results.seg.map50}\n")
            f.write(f"mAP@75: {train_results.seg.map75}\n")
            f.write(f"mAP@50-95: {train_results.seg.map}\n")
            f.write(f"mAP per class: {train_results.seg.maps}\n")
            f.write(f"Results class person: {train_results.seg.class_result(0)}\n")
            f.write(f"Results class car: {train_results.seg.class_result(1)}\n")

        print(f" Results saved in {results_file}")
        # Initialize the YOLO model
        # model = YOLO(MODEL_PATH)

        # # Define search space
        # search_space = {
        #    "lr0": (1e-5, 1e-2),
        #     "hsv_s": (0.0, 0.2),
        #     "cls":(0.5,2.0),
        #     "translate":(0.0,0.4),
        #     "scale":(0.0,0.5),
        #     "copy_paste":(0.0,0.4)
        # }

        # # Tune hyperparameters on COCO8 for 30 epochs
        # model.tune(
        #     data=DATASET_CONFIG_FINETUNE,
        #     epochs=20,
        #     iterations=100,
        #     optimizer="AdamW",
        #     space=search_space,
        #     plots=True,
        #     save=True,
        #     val=True,
        #     freeze=["backbone"],
        # )
        


if __name__ == "__main__":
    # Check if ultralytics is installed
    try:
        import ultralytics
    except ImportError:
        print("Ultralytics is not installed. Run:")
        print("pip install ultralytics")
        sys.exit(1)

    # Argument parser to select the mode and model name
    parser = argparse.ArgumentParser(description="Run YOLO inference or evaluation.")
    parser.add_argument("--mode", choices=["inference", "eval","train"], required=True, help="Choose whether to run inference or evaluation: 'inference' or 'eval'.")
    parser.add_argument("--model_name", type=str, default="yolov8n", help="Specify the YOLO model name (default: 'yolov8n').")
    args = parser.parse_args()

    # Assign model name and paths dynamically
    MODEL_NAME = args.model_name # YOLO11: yolo11x ; YOLO8: yolov8n
    MODEL_PATH = f"{MODEL_NAME}.pt"
    print(MODEL_PATH)
    
    DATASET_DIR = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo_masks_finetune"  # Input dataset directory
    OUTPUT_DIR = f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/outputs/output_{MODEL_NAME}_{args.mode}_new"   # Directory where results will be saved

    # Configuration inference
    BATCH_SIZE = 32  # Number of images to process per batch
    # Configuration eval
    DATASET_CONFIG = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo_finetune_eval.yaml"
    DATASET_CONFIG_FINETUNE = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo_finetune.yaml"

    # Run the selected mode
    if args.mode == "inference":
        yolo_run_inference(MODEL_PATH,DATASET_DIR,OUTPUT_DIR,BATCH_SIZE)
    elif args.mode == "eval":
        yolo_run_eval(MODEL_PATH,DATASET_CONFIG)
    elif args.mode =='train':
        yolo_run_train(MODEL_PATH,DATASET_CONFIG_FINETUNE)