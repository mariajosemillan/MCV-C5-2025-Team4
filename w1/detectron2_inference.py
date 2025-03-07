import os
import cv2
import torch
from detectron2 import config, model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def setup_predictor():
    '''Sets up the predictor with the model configuration and weights.

    This function loads the model configuration, sets the weights for the pre-trained model,
    sets the confidence threshold for inference, and specifies the device (CPU or GPU). 
    It then returns a DefaultPredictor instance for performing predictions and the configuration used.

    Returns:
        DefaultPredictor: The predictor object initialized with the configuration.
        cfg (CfgNode): The configuration object used to set up the model.
    '''
    cfg = config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)  # Pre-trained model weights
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_WEIGHTS) # Uncomment to use custom model weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold for inference
    cfg.MODEL.DEVICE = "cuda"
    
    predictor = DefaultPredictor(cfg)
    
    return predictor, cfg

def process_images_in_directory(predictor, cfg, input_dir, output_dir, subdir):
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
        selected_indices = torch.isin(instances.pred_classes, torch.tensor(CLASSES_TO_DETECT))
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

def process_all_images():
    '''Processes all images in the specified directories (['train', 'val', 'test']).

    This function initializes the predictor and configuration, then processes all images
    in the specified directories by calling `process_images_in_directory` for each.

    Returns:
        None
    '''
    predictor, cfg = setup_predictor()

    # Define the directories to process
    subdirs = ["train", "val"] # ["test"]

    # Process images in each subdirectory
    for subdir in subdirs:
        process_images_in_directory(predictor, cfg, INPUT_DIR, OUTPUT_DIR, subdir)

    print(f"Inference completed. Results in folder: '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "faster_rcnn_R_50_FPN_3x"
    MODEL_CONFIG = f"COCO-Detection/{MODEL_NAME}.yaml"  # Model configuration (you can change this depending on your model)
    # MODEL_WEIGHTS = "model_final_f10217.pkl"  # Path to the pre-trained model weights file or your custom model
    INPUT_DIR = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo/images"  # Path to your dataset with 'train' and 'val' folders
    OUTPUT_DIR = f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/output_{MODEL_NAME}"  # Folder where you will save the results

    # COCO class indices: 0 = person, 2 = car
    CLASSES_TO_DETECT = [0, 2]  # Only detect people (0) and cars (2)

    # Run inference on all images in the dataset
    process_all_images()
