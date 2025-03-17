import os
import sys
import json
import cv2
import torch
import argparse
import numpy as np
from detectron2 import config, model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from pycocotools.mask import encode
from pycocotools.coco import COCO
from detectron2.structures import Instances, Boxes

from utils import *

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
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_weights) # Change to path to .pth for custom model weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold  # Confidence threshold for inference
    cfg.MODEL.DEVICE = "cuda"

    predictor = DefaultPredictor(cfg)
    
    return predictor, cfg

def setup_trainer(model_name, model_weights, 
                  dataset_train, dataset_val, dataset_test, 
                  freeze_at, freeze_fpn=False, freeze_rpn=False, freeze_roi=False, 
                  lr=1e-4, optimizer="SGD", max_iters=5000,
                  output_dir='./output/'):
    cfg = config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_weights) # Change to path to .pth for custom model weights

    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.DATASETS.TRAIN = (dataset_train,)
    cfg.DATASETS.VAL = (dataset_val,)
    cfg.DATASETS.TEST = (dataset_test,)

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 4
    
    cfg.SOLVER.CHECKPOINT_PERIOD = max_iters

    cfg.SOLVER.OPTIMIZER = optimizer  # Opción "SGD" o "AdamW"

    # Configuración de parámetros para SGD
    cfg.SOLVER.BASE_LR = lr
    if optimizer=="SGD":
        cfg.SOLVER.MOMENTUM = 0.9  # Solo para SGD
        cfg.SOLVER.WEIGHT_DECAY = 0.0001
    
    cfg.SOLVER.MAX_ITER = max_iters
    cfg.SOLVER.STEPS = []
    

    cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at

    # Control manual de congelación
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False
    
    # Cargar el modelo temporalmente para modificar capas
    model = DefaultTrainer.build_model(cfg)

    if freeze_fpn:
        freeze_module(model.backbone.fpn_lateral2)
        freeze_module(model.backbone.fpn_output2)
        freeze_module(model.backbone.fpn_lateral3)
        freeze_module(model.backbone.fpn_output3)
        freeze_module(model.backbone.fpn_lateral4)
        freeze_module(model.backbone.fpn_output4)
        freeze_module(model.backbone.fpn_lateral5)
        freeze_module(model.backbone.fpn_output5)
        freeze_module(model.backbone.top_block)
    if freeze_rpn:
        freeze_module(model.proposal_generator)
    if freeze_roi:
        freeze_module(model.roi_heads)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # Batch size for ROI proposals

    # weight_person = 7016/19770
    # weight_car = 12754/19770
    # cfg.MODEL.ROI_HEADS.LOSS_WEIGHT = [weight_person, weight_car]

    # cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 2.0  # Aumenta el peso de la pérdida de regresión de cajas
    # cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1  # Reduce la pérdida de regresión del RPN

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 7 for aquarium domain shift
    
    cfg.TEST.EVAL_PERIOD = max_iters

    return cfg

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
    coco_predictions = []  # Lista para almacenar predicciones en formato COCO
    for image_id, image_file in enumerate(image_files, start=1):
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

            # Extract info for COCO JSON
            pred_boxes = selected_instances.pred_boxes.tensor.numpy()
            pred_classes = selected_instances.pred_classes.numpy()
            scores = selected_instances.scores.numpy()

            # If there are segmentations, convert to RLE
            if selected_instances.has("pred_masks"):
                pred_masks = selected_instances.pred_masks.numpy()
                rle_masks = [ 
                            {"size": rle["size"], "counts": rle["counts"].decode("utf-8")}  # Decode bytes to string
                            for rle in [encode(np.asfortranarray(mask.astype(np.uint8))) for mask in pred_masks]
                        ]
            else:
                rle_masks = [None] * len(pred_classes)  # If no masks, assign None

            # Save predictions in COCO format
            for i in range(len(pred_classes)):
                x, y, w, h = pred_boxes[i]
                coco_predictions.append({
                    "image_id": image_id,
                    "category_id": int(pred_classes[i]),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": float(scores[i]),
                    "segmentation": rle_masks[i]  # Save in format RLE
                })
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
    
    # Save JSON file with predictions
    json_output_path = os.path.join(output_dir, "predictions.json")
    with open(json_output_path, "w") as f:
        json.dump(coco_predictions, f, indent=4)

    print(f"Inference completed. Results in folder: '{output_dir}'.")
    print(f"Predictions saved in COCO format at: {json_output_path}")

def register_my_dataset(dataset_name, json_path, images_path, run_mode):
    '''Registers a dataset for use with Detectron2 using COCO format annotations.

    Args:
        dataset_name (str): Name of the dataset to register.
        json_path (str): Path to the COCO format JSON annotation file.
        images_path (str): Path to the directory containing dataset images.
    '''
    # Register the dataset using a COCO JSON annotation file
    register_coco_instances(dataset_name, {}, json_path, images_path)
    if run_mode=="finetune":
        metadata = MetadataCatalog.get(dataset_name)
        metadata.thing_classes = ["person", "car"] # FOR FINETUNE USE THIS
        # metadata.thing_classes = ['fish', 'reefs', 'aquatic plants', 'wrecks/ruins', 'human divers', 'robots', 'sea-floor']
        print(f"Classes in the dataset: {metadata.thing_classes}")

def detectron2_run_eval(model_name, model_weights, dataset_dir_json, dataset_dir_images, conf_threshold=0.5, dataset_name="dataset_val", output_dir="./output/"):
    '''Evaluates a Detectron2 model using a COCO-format dataset and saves results to a text file.

    Args:
        model_name (str): Name of the model configuration from Detectron2 model zoo.
        model_weights (str): Path to the pre-trained model weights.
        dataset_dir_json (str): Path to the dataset's COCO JSON annotation file.
        dataset_dir_images (str): Path to the dataset's image directory.
        conf_threshold (float, optional): Confidence threshold for model predictions. Defaults to 0.5.
        dataset_name (str, optional): Name to register the dataset under. Defaults to "dataset_val".
        output_dir (str, optional): Directory where evaluation results will be saved. Defaults to "./output/".
    '''
    # Dataset configuration -- register dataset
    register_my_dataset(dataset_name=dataset_name, json_path=dataset_dir_json, images_path=dataset_dir_images, run_mode="eval")  # Register the dataset before evaluation
    
    # Load model configuration
    cfg = config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))  # Load pretrained model config
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_weights) # Load model weights
    cfg.DATASETS.TEST = (dataset_name,)  # Set evaluation dataset
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold  # Confidence threshold for detections

    # Initialize the predictor with the model configuration
    predictor = DefaultPredictor(cfg)

    # Initialize evaluation components
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=output_dir)
    test_loader = build_detection_test_loader(cfg, dataset_name)

    # Perform inference and evaluation
    results = inference_on_dataset(predictor.model, test_loader, evaluator)
    print(results)
    # Save evaluation results to a txt file
    results_txt_path = f"{output_dir}/results.txt"
    with open(results_txt_path, "w") as f:
        f.write(str(results))
    print(f"Results saved to {results_txt_path}")

def train(cfg, dataset_val=None):
    # trainer = DefaultTrainer(cfg)
    trainer = MyTrainer(cfg)

    # Create a custom validation loss object
    val_loss = ValidationLoss(cfg)

    # Register the custom validation loss object as a hook to the trainer
    trainer.register_hooks([val_loss])

    # Swap the positions of the evaluation and checkpointing hooks so that the validation loss is logged correctly
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    # Resume training from a checkpoint or load the initial model weights
    trainer.resume_or_load(resume=False)

    trainer.train()

def detectron2_run_finetune(model_name, model_weights, 
                            dataset_train_json, dataset_train_images, 
                            dataset_val_json, dataset_val_images, 
                            dataset_test_json,dataset_test_images, 
                            output_dir='./output/'):
    # Register the datasets using a COCO JSON annotation file
    dataset_train = "dataset_train"
    register_my_dataset(dataset_train, json_path=dataset_train_json, images_path=dataset_train_images, run_mode="finetune")
    
    dataset_val = "dataset_val"
    register_my_dataset(dataset_val, json_path=dataset_val_json, images_path=dataset_val_images, run_mode="finetune")
    
    dataset_test = "dataset_test"
    register_my_dataset(dataset_test, json_path=dataset_test_json, images_path=dataset_test_images, run_mode="finetune")

    experiments = [
        # (5, True, True, True, 1e-4, "SGD", 10000),   # Exp
        # (5, True, True, False),  # Exp
        (5, True, False, False, 1e-4, "SGD", 2000), # Exp
        # (5, False, False, False, 1e-4, "SGD"),# Exp1
        # (4, False, False, False, 1e-4, "SGD"),# Exp2
        # (3, False, False, False, 1e-4, "SGD", 20000),# Exp3
        # (3, False, False, False, 1e-5, "SGD", 10000),# Exp4
        # (3, False, False, False, 1e-3, "SGD"),# Exp5
        # (2, False, False, False, 1e-4, "SGD"),# Exp6
        # (0, False, False, False, 1e-4, "SGD") # Exp7
    ]
    for exp_id, (freeze_at, freeze_fpn, freeze_rpn, freeze_roi, lr, optimizer, max_iters) in enumerate(experiments, start=1):
        output_dir_exp = get_next_experiment_folder(output_dir)
        print(f"Running Experiment {exp_id}: FREEZE_AT={freeze_at}, FPN={'🔒' if freeze_fpn else '🔓'}, RPN={'🔒' if freeze_rpn else '🔓'}, ROI={'🔒' if freeze_roi else '🔓'}")
        cfg = setup_trainer(model_name, model_weights, 
                            dataset_train, dataset_val, dataset_test, 
                            freeze_at, freeze_fpn, freeze_rpn, freeze_roi, 
                            lr=lr, optimizer=optimizer, max_iters=max_iters,
                            output_dir=output_dir_exp)
        train(cfg, dataset_val)
        print(f"Experiment {exp_id} completed. Results saved in {output_dir_exp}\n")

        # dataset_name = "dataset_test"
        # register_my_dataset(dataset_name, json_path="/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/test_coco_car1_finetune_test.json", images_path="/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo_finetune/images/test")  # Register the dataset before evaluation
        # cfg.DATASETS.TEST = (dataset_name,)

        # predictor = DefaultPredictor(cfg)
        # input_dir = f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo_masks_finetune/images/test"
        # input_dir = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/aquarium_segmentation/test"
        # output_dir = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/outputs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_aquarium_best_model"
        # os.makedirs(output_dir, exist_ok=True)
        # detectron2_run_inference(predictor, cfg, input_dir, output_dir, [1, 2, 3, 4, 5, 6, 7])

if __name__ == "__main__":
    # Argument parser to select the mode and model name
    parser = argparse.ArgumentParser(description="Run DETECTRON2 inference or evaluation.")
    parser.add_argument("--run_mode", choices=["inference", "eval", "finetune"], required=True, 
                        help="Choose the mode to run: 'inference' for making predictions on images, 'eval' for evaluating the model performance, or 'finetune' for training the model on a custom dataset.")
    parser.add_argument("--model_config", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
                        help="Specify the model configuration file from Detectron2 Model Zoo (default: 'mask_rcnn_R_50_FPN_3x').")
    parser.add_argument("--model_weights", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
                        help="Specify the model weights file (default: 'mask_rcnn_R_50_FPN_3x').")
    parser.add_argument("--dataset_dir", type=str, default="ghome/c5mcv04/MCV-C5-2025-Team4/dataset", 
                        help="Path to the dataset directory containing images and annotations.")
    parser.add_argument("--json_path", type=str, 
                        help="Path to the COCO format JSON annotation file (required for 'eval' mode).")
    parser.add_argument("--json_dir", type=str, 
                        help=" (required for 'finetune' mode)")
    parser.add_argument("--output_dir", type=str, default="ghome/c5mcv04/MCV-C5-2025-Team4/dataset/output", 
                        help="Path to the output directory where results will be stored.")
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 1], 
                        help="List of class IDs to detect (default: [0,1], not used in 'eval' mode).")
    parser.add_argument("--conf_threshold", type=float, default=0.5, 
                        help="Confidence threshold for object detection (default: 0.5).")
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

    OUTPUT_DIR = f"{args.output_dir}/outputs/{MODEL_NAME}_{RUN_MODE}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n✅ Results will be saved in: {OUTPUT_DIR}\n")

    if RUN_MODE == "inference":
        INPUT_DIR = f"{DATASET_DIR}"  # Path to the dataset with 'train' and 'val' folders
    
        detectron2_run_inference(model_name=MODEL_CONFIG, model_weights=MODEL_WEIGHTS, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, classes=CLASSES_TO_DETECT) # Run inference on all images in the dataset
    
    elif RUN_MODE == "eval":
        INPUT_DIR_IMAGES = f"{DATASET_DIR}"
        
        # Check if json_path argument was provided
        if not args.json_path:
            print("Error: --json_path is required when --run_mode is 'eval'.", file=sys.stderr)
            sys.exit(1)
        INPUT_DIR_JSON = args.json_path
        
        CONF_THRESHOLD = args.conf_threshold
        
        detectron2_run_eval(model_name=MODEL_CONFIG, model_weights=MODEL_WEIGHTS, dataset_dir_images=INPUT_DIR_IMAGES, dataset_dir_json=INPUT_DIR_JSON, conf_threshold=CONF_THRESHOLD, dataset_name="dataset_val", output_dir=OUTPUT_DIR)
    
    elif RUN_MODE == "finetune":
        
        # Define paths for training, validation, and test images
        INPUT_DIR_IMAGES_TRAIN = os.path.join(DATASET_DIR, "train")
        INPUT_DIR_IMAGES_VAL = os.path.join(DATASET_DIR, "val")
        INPUT_DIR_IMAGES_TEST = os.path.join(DATASET_DIR, "test")

        # Check if json_dir argument was provided    
        if not args.json_dir:
            print("Error: --json_dir is a required argument for 'finetune' mode.", file=sys.stderr)
            sys.exit(1)

        # Verify if json_dir is a valid directory
        if not os.path.isdir(args.json_dir):
            print(f"Error: The directory '{args.json_dir}' does not exist or is not a valid directory.", file=sys.stderr)
            sys.exit(1)
        
        # Define paths for the training, validation, and test JSON files
        INPUT_DIR_JSON = args.json_dir
        INPUT_DIR_JSON_TRAIN = os.path.join(INPUT_DIR_JSON, "train.json")
        INPUT_DIR_JSON_VAL = os.path.join(INPUT_DIR_JSON, "val.json")
        INPUT_DIR_JSON_TEST = os.path.join(INPUT_DIR_JSON, "test.json")

        # Run the fine-tuning process
        detectron2_run_finetune(model_name=MODEL_CONFIG, model_weights=MODEL_WEIGHTS, 
                                dataset_train_json=INPUT_DIR_JSON_TRAIN, dataset_train_images=INPUT_DIR_IMAGES_TRAIN, 
                                dataset_val_json=INPUT_DIR_JSON_VAL, dataset_val_images=INPUT_DIR_IMAGES_VAL, 
                                dataset_test_json=INPUT_DIR_JSON_TEST, dataset_test_images=INPUT_DIR_IMAGES_TEST, 
                                output_dir=OUTPUT_DIR)