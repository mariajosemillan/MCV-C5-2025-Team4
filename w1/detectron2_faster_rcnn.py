import os
import cv2
import torch
import argparse
from detectron2 import config, model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, verify_results

def setup_predictor():
    '''Sets up the predictor with the model configuration and weights.

    This function loads the model configuration, sets the weights for the pre-trained model,
    sets the confidence threshold for inference, and specifies the device (CPU or GPU). 
    It then returns a DefaultPredictor instance for performing predictions and the configuration used.

    Returns:
        tuple: A tuple containing:
            - DefaultPredictor: The predictor object initialized with the configuration.
            - CfgNode: The configuration object used to set up the model.
    '''
    cfg = config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)  # Pre-trained model weights
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_WEIGHTS) # Uncomment to use custom model weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold for inference
    cfg.MODEL.DEVICE = "cuda"
    
    predictor = DefaultPredictor(cfg)
    
    return predictor, cfg

def detectron2_run_inference_in_directory(predictor, cfg, input_dir, output_dir, subdir, classes):
    '''Processes all images in a given directory, performs inference using the predictor,
    and saves the resulting images with predictions to the output directory.

    This function reads all image files from the specified input directory, applies the model 
    to each image, filters the predicted instances for the specified classes, and saves the 
    images with drawn predictions to the corresponding output directory.

    Args:
        predictor (DefaultPredictor): The predictor used to perform inference on the images.
        cfg (CfgNode): The configuration object used to set up the model and inference parameters.
        input_dir (str): The path to the directory containing the input images.
        output_dir (str): The path to the directory where the output images will be saved.
        subdir (str): The subdirectory within the input and output directories to process (e.g., 'train' or 'val').
        classes (list): List of class IDs to filter detections.

    Returns:
        None
    '''
    input_folder = os.path.join(input_dir, subdir)
    output_folder = os.path.join(output_dir, subdir)
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in the folder
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Processing {len(image_files)} images in {subdir}...")

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
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
            v = v.draw_instance_predictions(selected_instances)
            result_img = v.get_image()[:, :, ::-1]
        else:
            result_img = img  # If no predictions, just save the original image


        # Save the resulting image with predictions
        output_path = os.path.join(output_folder, os.path.basename(image_file))
        cv2.imwrite(output_path, result_img)

        print(f"Image processed: {output_path}")

def detectron2_run_inference(input_dir, output_dir, classes):
    '''Processes all images in the specified directories (['train', 'val', 'test']).

    This function initializes the predictor and configuration, then processes all images
    in the specified directories by calling `process_images_in_directory` for each.

    Args:
        input_dir (str): Path to the input dataset directory.
        output_dir (str): Path to save processed images with predictions.
        classes (list): List of class IDs to detect.

    Returns:
        None
    '''
    predictor, cfg = setup_predictor()

    # Define the directories to process
    subdirs = ["train", "val"] # ["test"]

    # Process images in each subdirectory
    for subdir in subdirs:
        detectron2_run_inference_in_directory(predictor, cfg, input_dir, output_dir, subdir, classes=classes)

    print(f"Inference completed. Results in folder: '{output_dir}'.")

def detectron2_run_eval(config_file="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", weights_file="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", output_dir="./output/"):
    '''Evaluates a Faster R-CNN model using the COCO dataset.

    Args:
        config_file (str, optional): Path to the model configuration file. 
                                     Defaults to "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml".
        weights_file (str, optional): Path to the model weights file.
                                      Defaults to "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml".
        output_dir (str, optional): Directory where evaluation results will be saved.
                                    Defaults to "./output/".

    Returns:
        None
    '''
    # Dataset configuration
    dataset_name = "dataset_val"
    register_my_dataset(dataset_name)  # Register the dataset before evaluation

    # Load model configuration
    cfg = config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))  # Load pretrained model config
    # cfg.MODEL.WEIGHTS = weights_file  # Especificar los pesos del modelo entrenado
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") # Load model weights
    cfg.DATASETS.TEST = (f"{dataset_name}",)  # Set evaluation dataset
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold for detections

    # Initialize the predictor with the model configuration
    predictor = DefaultPredictor(cfg)

    # Evaluate the model on the validation set
    evaluator = COCOEvaluator("dataset_val", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "dataset_val")
    # Run inference on the dataset
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    # Print evaluation results (mAP, precision, recall, etc.)
    print(results["bbox"])


def register_my_dataset(dataset_name):
    '''Registers a COCO-format dataset for evaluation.

    Args:
        dataset_name (str): Name of the dataset to be registered.

    Returns:
        None
    '''
    # Register the dataset using a COCO JSON annotation file
    register_coco_instances(f"{dataset_name}", {}, "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/val_coco.json", "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo/images/val")
    
    metadata = MetadataCatalog.get("dataset_val")
    # Assign class names (must match the dataset annotations)
    metadata.thing_classes = MetadataCatalog.get("coco_2017_val").thing_classes
    print(f"Classes in the dataset: {metadata.thing_classes}")

if __name__ == "__main__":

    # Argument parser to select the mode and model name
    parser = argparse.ArgumentParser(description="Run DETECTRON2 inference or evaluation.")
    parser.add_argument("--mode", choices=["inference", "eval"], required=True, help="Choose whether to run inference or evaluation: 'inference' or 'eval'.")
    parser.add_argument("--model_name", type=str, default="faster_rcnn_R_50_FPN_3x", help="Specify the model name (default: 'faster_rcnn_R_50_FPN_3x').")
    args = parser.parse_args()

    # Configuration
    MODEL_NAME = args.model_name
    MODEL_CONFIG = f"COCO-Detection/{MODEL_NAME}.yaml"  # Model configuration
    # MODEL_WEIGHTS = "model_final_f10217.pkl"  # Path to the pre-trained model weights file a custom model
    INPUT_DIR = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo/images"  # Path to the dataset with 'train' and 'val' folders
    OUTPUT_DIR = f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/output_{MODEL_NAME}_eval"  # Folder where results will be saved

    CLASSES_TO_DETECT = [0, 2] # COCO class indices: 0 = person, 2 = car


    if args.mode == "inference":
        detectron2_run_inference(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, classes=CLASSES_TO_DETECT) # Run inference on all images in the dataset
    elif args.mode == "eval":
        weights_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        detectron2_run_eval(weights_file=weights_path, output_dir=OUTPUT_DIR)
