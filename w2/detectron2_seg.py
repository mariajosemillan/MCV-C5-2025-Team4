import os
import cv2
import torch
import argparse
import numpy as np
from detectron2 import config, model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def setup_predictor(model_name, model_weights, conf_threshold=0.5):
    '''Sets up the Detectron2 predictor with the specified model configuration and weights.

    This function loads the model configuration, sets the weights for the pre-trained model,
    sets the confidence threshold for inference, and specifies the device (CPU or GPU). 
    It then returns a DefaultPredictor instance for performing predictions and the configuration used.

    Args:
        model_name (str): The name of the model configuration file from Detectron2's model zoo.
        model_weights (str): The corresponding pre-trained weights file for the model.
        conf_threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.

    Returns:
        tuple: A tuple containing:
            - DefaultPredictor: The predictor object initialized with the configuration.
            - CfgNode: The configuration object used to set up the model.
    '''
    cfg = config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold  # Confidence threshold for inference
    cfg.MODEL.DEVICE = "cuda"

    predictor = DefaultPredictor(cfg)
    
    return predictor, cfg

def detectron2_run_inference(model_name, model_weights, input_dir, output_dir, classes):
    '''Runs inference on images in the specified input directory using Detectron2.
    
    Args:
        model_name (str): The name of the model configuration file.
        model_weights (str): The corresponding model weights.
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where processed images will be saved.
        classes (list[int]): List of class IDs to filter predictions.
    
    Returns:
        None
    '''
    predictor, cfg = setup_predictor(model_name, model_weights)

    # Get all image files in the folder
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images in {input_dir}...")

    for image_file in image_files:
        # Load the image
        img = cv2.imread(image_file)

        # Perform inference
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        # Filter instances for the classes of interest
        selected_indices = torch.isin(instances.pred_classes, torch.tensor(classes))
        # Select instances that belong to the desired classes
        selected_instances = instances[selected_indices]
        if selected_instances.has("pred_boxes") and selected_instances.has("pred_classes"):
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1)
            v = v.draw_instance_predictions(selected_instances)
            result_img = v.get_image()[:, :, ::-1]
        else:
            result_img = img  # If no predictions, just save the original image
            print(f"No predictions for image {image_file}, saving original image.")

        # Save the resulting image with predictions
        output_path = os.path.join(output_dir, os.path.basename(image_file))
        result = cv2.imwrite(output_path, result_img)
        if result:
            print(f"Image {image_file} processed and saved at: {output_path}")
        else:
            print(f"Error saving processed {image_file}. Output dir selected: {output_path}")

    print(f"Inference completed. Results in folder: '{output_dir}'.")

if __name__ == "__main__":

    # Argument parser to select the mode and model name
    parser = argparse.ArgumentParser(description="Run DETECTRON2 inference or evaluation.")
    parser.add_argument("--run_mode", choices=["inference", "eval", "finetune"], required=True, help="Choose whether to run inference or evaluation: 'inference' or 'eval'.")
    parser.add_argument("--model_config", type=str, default="CCOCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="Specify the model config (default: 'faster_rcnn_R_50_FPN_3x').")
    parser.add_argument("--model_weights", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="Specify the model weights (default: 'faster_rcnn_R_50_FPN_3x').")
    parser.add_argument("--dataset_dir", type=str, default="ghome/c5mcv04/MCV-C5-2025-Team4/dataset", help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, default="ghome/c5mcv04/MCV-C5-2025-Team4/dataset/output", help="Path to the dataset directory.")
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 1], help="List of class IDs to detect (default: [0,1]).")
    args = parser.parse_args()

    RUN_MODE = args.run_mode

    # Model config
    MODEL_CONFIG = args.model_config  # Model configuration
    try:
        config_path = model_zoo.get_config_file(MODEL_CONFIG)
        print("Model config path:", config_path)
    except RuntimeError as e:
        print("Error:", e)

    MODEL_NAME = os.path.splitext(MODEL_CONFIG)[0]
    MODEL_WEIGHTS = args.model_weights
    CLASSES_TO_DETECT = args.classes # COCO class indices: 0 = person, 2 = car
    
    # Set up directories
    DATASET_DIR = args.dataset_dir
    # Check if dataset directory exists
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Error: The dataset directory '{DATASET_DIR}' does not exist. Please provide a valid path.")

    OUTPUT_DIR = f"{args.output_dir}/{MODEL_NAME}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n✅ Results will be saved in: {OUTPUT_DIR}\n")

    if RUN_MODE == "inference":
        INPUT_DIR = f"{DATASET_DIR}"  # Path to the dataset with 'train' and 'val' folders
        detectron2_run_inference(model_name=MODEL_CONFIG, model_weights=MODEL_WEIGHTS, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, classes=CLASSES_TO_DETECT) # Run inference on all images in the dataset