import os
import re
import cv2
import torch
import argparse
import numpy as np
from detectron2 import config, model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, verify_results

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import hooks
from detectron2.utils.events import EventStorage
import detectron2.utils.comm as comm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils

from detectron2.structures import Instances, Boxes

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
    # cfg = config.get_cfg()
    # # cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    # cfg.merge_from_file("/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/output_faster_rcnn_R_50_FPN_3x_finetune_SGD/_exp3/config.yaml")
    # # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)  # Pre-trained model weights
    # MODEL_WEIGHTS = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/output_faster_rcnn_R_50_FPN_3x_finetune_SGD/_exp3/model_final.pth"
    # cfg.MODEL.WEIGHTS = MODEL_WEIGHTS # Uncomment to use custom model weights
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold for inference
    # cfg.MODEL.DEVICE = "cuda"

    dataset_train_json_path = f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/train_coco_car1_finetune_2.json"
    dataset_train_images_path = f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo_finetune_2/images/train"
    dataset_val_json_path = f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/val_coco_car1_finetune_2.json"
    dataset_val_images_path = f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo_finetune_2/images/val"
    dataset_train = "dataset_train"
    register_my_dataset(dataset_train, json_path=dataset_train_json_path, images_path=dataset_train_images_path)  # Register the dataset before evaluation
    dataset_val = "dataset_val"
    register_my_dataset(dataset_val, json_path=dataset_val_json_path, images_path=dataset_val_images_path)  # Register the dataset before evaluation

    cfg = setup_training(MODEL_CONFIG, MODEL_CONFIG, dataset_train, dataset_val, freeze_at=3, freeze_fpn=False, freeze_rpn=False, freeze_roi=False, lr=1e-4, optimizer="SGD", max_iters=5000)
    cfg.MODEL_WEIGHTS = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/output_faster_rcnn_R_50_FPN_3x_finetune_SGD/_exp3/model_final.pth"
    dataset_name = "dataset_test"
    register_my_dataset(dataset_name, json_path="/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/test_coco_car1_finetune_test.json", images_path="/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo_finetune/images/test")  # Register the dataset before evaluation
    cfg.DATASETS.TEST = (dataset_name,)

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
        print(outputs)
        instances = outputs["instances"].to("cpu")
        # selected_instances = instances[0]
        # Filter instances for the classes of interest
        selected_indices = torch.isin(instances.pred_classes, torch.tensor(classes))
        # Select instances that belong to the desired classes
        selected_instances = instances[selected_indices]
        print(selected_indices)
        print(instances.pred_classes)
        if selected_instances.has("pred_boxes") and selected_instances.has("pred_classes"):
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1)
            v = v.draw_instance_predictions(selected_instances)
            result_img = v.get_image()[:, :, ::-1]
        else:
            result_img = img  # If no predictions, just save the original image
            print("NO PREDICTIONS, SAVING ORIGINAL IMAGE")


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
    # subdirs = ["train", "val"] # ["test"]
    subdirs = ["test"]

    # Process images in each subdirectory
    for subdir in subdirs:
        detectron2_run_inference_in_directory(predictor, cfg, input_dir, output_dir, subdir, classes=classes)

    print(f"Inference completed. Results in folder: '{output_dir}'.")

def detectron2_run_eval(dataset_dir_json, dataset_dir_images, config_file="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", weights_file="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", output_dir="./output/"):
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
    register_my_dataset(dataset_name, json_path=dataset_dir_json, images_path=dataset_dir_images)  # Register the dataset before evaluation

    # Load model configuration
    cfg = config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))  # Load pretrained model config
    # cfg.MODEL.WEIGHTS = weights_file  # Especificar los pesos del modelo entrenado
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file) # Load model weights
    cfg.MODEL.WEIGHTS = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/output_faster_rcnn_R_50_FPN_3x_finetune_SGD/_exp3/model_final.pth"
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


def register_my_dataset(dataset_name, json_path, images_path):
    '''Registers a COCO-format dataset for evaluation.

    Args:
        dataset_name (str): Name of the dataset to be registered.

    Returns:
        None
    '''
    # Register the dataset using a COCO JSON annotation file
    register_coco_instances(f"{dataset_name}", {}, json_path, images_path)
    
    metadata = MetadataCatalog.get("dataset_val")
    # Assign class names (must match the dataset annotations)
    # metadata.thing_classes = MetadataCatalog.get("coco_2017_val").thing_classes
    # Assign classes manually
    metadata.thing_classes = ["pedestrian", "car"]
    print(f"Classes in the dataset: {metadata.thing_classes}")

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluator")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    # @classmethod
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg, mapper=AlbumentationsMapper())

def setup_training(model_cfg, model_weights, dataset_train, dataset_val, freeze_at, freeze_fpn=False, freeze_rpn=False, freeze_roi=False, lr=0.001, optimizer="AdamW", max_iters=5000, output_dir='./output/'):
    cfg = config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"{model_cfg}"))

    cfg.DATASETS.TRAIN = (dataset_train,)
    cfg.DATASETS.VAL = (dataset_val,)
    cfg.DATASETS.TEST = (dataset_val,)

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATALOADER.NUM_WORKERS = 4

    # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    # cfg.DATALOADER.AUGMENTATIONS = transform

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{model_weights}") # Load pre-trained weights

    cfg.SOLVER.IMS_PER_BATCH = 4
    
    cfg.SOLVER.CHECKPOINT_PERIOD = max_iters//10

    # Configuración de parámetros para SGD
    cfg.SOLVER.BASE_LR = lr

    if optimizer == "SGD":
        cfg.SOLVER.MOMENTUM = 0.9  # Solo para SGD
        cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.OPTIMIZER = optimizer
    
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

    weight_person = 1 / 7016/19767
    weight_car = 1 / 12751/19767
    cfg.MODEL.ROI_HEADS.LOSS_WEIGHT = [weight_person, weight_car]

    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 2.0  # Aumenta el peso de la pérdida de regresión de cajas
    cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1  # Reduce la pérdida de regresión del RPN

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    
    cfg.TEST.EVAL_PERIOD = max_iters//10

    # Guardar configuración en un archivo txt
    config_path = os.path.join(cfg.OUTPUT_DIR, "config.txt")
    with open(config_path, "w") as f:
        f.write(cfg.dump())  # Guardar la configuración completa en formato YAML

    return cfg

class ValidationLoss(HookBase):
    """
    A hook that computes validation loss during training.

    Attributes:
        cfg (CfgNode): The detectron2 config node.
        _loader (iterator): An iterator over the validation dataset.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): The detectron2 config node.
        """
        super().__init__()
        self.cfg = cfg.clone()
        # Switch to the validation dataset
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        # Build the validation data loader iterator
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        """
        Computes the validation loss after each training step.
        """
        # Get the next batch of data from the validation data loader
        data = next(self._loader)
        with torch.no_grad():
            # Compute the validation loss on the current batch of data
            loss_dict = self.trainer.model(data)

            # Check for invalid losses
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            # Reduce the loss across all workers
            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # Save the validation loss in the trainer storage
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced)


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

def detectron2_run_finetune(dataset_train_json, dataset_train_images, dataset_val_json, dataset_val_images, model_cfg="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", model_weights="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", output_dir='./output/'):
    # Register the datasets using a COCO JSON annotation file
    dataset_train = "dataset_train"
    register_my_dataset(dataset_train, json_path=dataset_train_json, images_path=dataset_train_images)
    
    dataset_val = "dataset_val"
    register_my_dataset(dataset_val, json_path=dataset_val_json, images_path=dataset_val_images)
    
    experiments = [
        # (5, True, True, True),   # Exp1
        # (5, True, True, False),  # Exp2
        # (5, True, False, False), # Exp3
        # (5, False, False, False),# Exp4
        # (4, False, False, False),# Exp5
        # (3, False, False, False, 0.01, "SGD", 5000),# Exp6
        # (3, False, False, False, 0.001, "SGD", 5000),# Exp6
        # (3, False, False, False, 0.0001, "SGD", 5000),# Exp6
        (3, False, False, False, 0.00001, "AdamW", 5000),# Exp6
        # (2, False, False, False),# Exp7
        # (0, False, False, False) # Exp8
    ]
    for exp_id, (freeze_at, freeze_fpn, freeze_rpn, freeze_roi, lr, optimizer, max_iters) in enumerate(experiments, start=1):
        output_dir_exp = get_next_experiment_folder(output_dir)
        print(f"Running Experiment {exp_id}: FREEZE_AT={freeze_at}, FPN={'🔒' if freeze_fpn else '🔓'}, RPN={'🔒' if freeze_rpn else '🔓'}, ROI={'🔒' if freeze_roi else '🔓'}")
        print(f"Other params: LR={lr}, OPTIMIZER={optimizer}, MAX_ITERS={max_iters}")
        cfg = setup_training(model_cfg, model_weights, dataset_train, dataset_val, freeze_at, freeze_fpn, freeze_rpn, freeze_roi, lr, optimizer, max_iters, output_dir_exp)
        train(cfg, dataset_val)
        print(f"Experiment {exp_id} completed. Results saved in {output_dir_exp}\n")

def get_next_experiment_folder(base_output_dir):
    os.makedirs(base_output_dir, exist_ok=True)
    existing_exps = [d for d in os.listdir(base_output_dir) if d.startswith("_exp")]
    exp_numbers = [int(d.replace("_exp", "")) for d in existing_exps if d.replace("_exp", "").isdigit()]
    next_exp = max(exp_numbers, default=0) + 1
    return os.path.join(base_output_dir, f"_exp{next_exp}")

if __name__ == "__main__":

    # Argument parser to select the mode and model name
    parser = argparse.ArgumentParser(description="Run DETECTRON2 inference or evaluation.")
    parser.add_argument("--mode", choices=["inference", "eval", "finetune"], required=True, help="Choose whether to run inference or evaluation: 'inference' or 'eval'.")
    parser.add_argument("--model_name", type=str, default="faster_rcnn_R_50_FPN_3x", help="Specify the model name (default: 'faster_rcnn_R_50_FPN_3x').")
    args = parser.parse_args()

    # Configuration
    MODEL_NAME = args.model_name
    MODEL_CONFIG = f"COCO-Detection/{MODEL_NAME}.yaml"  # Model configuration
    # MODEL_WEIGHTS = "model_final_f10217.pkl"  # Path to the pre-trained model weights file a custom model
    DATASET_DIR = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset"
    
    OUTPUT_DIR = f"{DATASET_DIR}/output_{MODEL_NAME}_{args.mode}_inf"
    # OUTPUT_DIR = get_next_experiment_folder(BASE_OUTPUT_DIR)
    # print(f"\n✅ Results will be saved in: {OUTPUT_DIR}\n")
    # OUTPUT_DIR = f"{DATASET_DIR}/output_{MODEL_NAME}_{args.mode}"  # Folder where results will be saved

    CLASSES_TO_DETECT = [0, 1] # COCO class indices: 0 = person, 2 = car

    if args.mode == "inference":
        INPUT_DIR = f"{DATASET_DIR}/dataset_yolo_finetune/images"  # Path to the dataset with 'train' and 'val' folders
        detectron2_run_inference(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, classes=CLASSES_TO_DETECT) # Run inference on all images in the dataset
    elif args.mode == "eval":
        INPUT_DIR = f"{DATASET_DIR}/dataset_yolo/images"  # Path to the dataset with 'train' and 'val' folders
        weights_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        # dataset_json_path = f"{DATASET_DIR}/val_coco_eval.json"
        dataset_json_path = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/test_coco_car1_finetune_test.json"
        # dataset_images_path = f"{DATASET_DIR}/dataset_yolo/images/val"
        dataset_images_path = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo_finetune/images/test"
        detectron2_run_eval(dataset_dir_json=dataset_json_path, dataset_dir_images=dataset_images_path, weights_file=weights_path, output_dir=OUTPUT_DIR)
    elif args.mode == "finetune":
        INPUT_DIR = f"{DATASET_DIR}/dataset_yolo_finetune_2/images"  # Path to the dataset with 'train' and 'val' folders
        # weights_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        weights_path = ""
        dataset_train_json_path = f"{DATASET_DIR}/train_coco_car1_finetune_2.json"
        dataset_train_images_path = f"{INPUT_DIR}/train"
        dataset_val_json_path = f"{DATASET_DIR}/val_coco_car1_finetune_2.json"
        dataset_val_images_path = f"{INPUT_DIR}/val"
        detectron2_run_finetune(dataset_train_json_path, dataset_train_images_path, dataset_val_json_path, dataset_val_images_path, model_cfg=MODEL_CONFIG, model_weights=MODEL_CONFIG, output_dir=OUTPUT_DIR)