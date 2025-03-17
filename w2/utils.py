import os
import cv2
import torch
import numpy as np

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import build_detection_train_loader, DatasetMapper, detection_utils
import detectron2.utils.comm as comm
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator

import albumentations as A
import pycocotools.mask as maskUtils

def print_config_models():
    config_dir = os.path.dirname(model_zoo.__file__) + "/configs/"
    print("Available model configs in Detectron2 Model Zoo:")
    for root, dirs, files in os.walk(config_dir):
        for file in files:
            if file.endswith(".yaml"):
                print(os.path.relpath(os.path.join(root, file), config_dir))
    
def get_next_experiment_folder(base_output_dir):
    os.makedirs(base_output_dir, exist_ok=True)
    existing_exps = [d for d in os.listdir(base_output_dir) if d.startswith("_exp")]
    exp_numbers = [int(d.replace("_exp", "")) for d in existing_exps if d.replace("_exp", "").isdigit()]
    next_exp = max(exp_numbers, default=0) + 1
    return os.path.join(base_output_dir, f"_exp{next_exp}")

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

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluator")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = AlbumentationsMapper(cfg, is_train=True, augmentations=get_augmentations())
        return build_detection_train_loader(cfg, mapper=mapper)
    
def rle_to_mask(rle, height, width):
    """Convierte una máscara RLE de COCO a una matriz binaria."""
    return maskUtils.decode(rle).reshape((height, width))

def mask_to_rle(mask):
    """Convierte una máscara binaria a formato RLE (si es necesario)."""
    return maskUtils.encode(np.asfortranarray(mask))

def mask_to_polygon(mask):
    """Convierte una máscara binaria a una lista de polígonos."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        # Convierte el contorno en una lista de tuplas (x, y)
        polygon = contour.reshape(-1, 2).tolist()
        polygons.append(polygon)
    return polygons

def get_augmentations() -> A.Compose:
	"""Get the augmentations to apply.

	Returns:
		A.Compose: Compose of augmentations from albumentations.
	"""
	return A.Compose([
        A.OneOf([
			# A.RandomCrop(width=1000, height=300),
			# A.RandomCrop(width=300, height=150),
		]),
        A.Illumination(p=0.5, intensity_range=(0.01, 0.2)),
        # A.AtLeastOneBBoxRandomCrop(height=200, width=800),
    ], bbox_params=A.BboxParams(format='coco', min_area=200, min_visibility=0.1, label_fields=['category_ids']))

class AlbumentationsMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True, augmentations=None):
        """Initializes albumentations mapper.

        Args:
            cfg (Any): Configuration for the model.
            is_train (bool, optional): Whether is train dataset. Defaults to True.
            augmentations (Any, optional): Augmentations from albumentations to apply. Defaults to None.
        """
        super().__init__(cfg, is_train, instance_mask_format="bitmask")
        self.augmentations = augmentations
        self.mask_format = cfg.INPUT.MASK_FORMAT

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = cv2.imread(dataset_dict["file_name"]) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_train and "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")

            # Filtrar bounding boxes inválidos (width = 0 o height = 0)
            valid_annotations = [
                obj for obj in annotations if obj["bbox"][2] > 0 and obj["bbox"][3] > 0
            ]

            if valid_annotations:  # Aplicar transformaciones solo si hay bboxes válidos
                bboxes = [obj["bbox"] for obj in valid_annotations]
                category_ids = [obj["category_id"] for obj in valid_annotations]
                
                transformed = self.augmentations(image=image, bboxes=bboxes, category_ids=category_ids)
                image = transformed["image"]
                
                # Update the bounding boxes with transformed coordinates
                for i, annotation in enumerate(valid_annotations):
                    if i < len(transformed["bboxes"]):
                        annotation["bbox"] = transformed["bboxes"][i]
                
                # Convert to Instances format for Detectron2
                annos = []
                for annotation in valid_annotations:
                    obj = {
                        "bbox": annotation["bbox"],
                        "bbox_mode": annotation.get("bbox_mode", BoxMode.XYWH_ABS),
                        "segmentation": annotation.get("segmentation"),
                        "category_id": annotation["category_id"],
                        "iscrowd": annotation.get("iscrowd", 0),
                    }
                    annos.append(obj)
                
                # Create Instances object with the correct image size
                instances = detection_utils.annotations_to_instances(annos, image.shape[:2], mask_format=self.mask_format)
                dataset_dict["instances"] = instances
        
        # Convert to CHW format
        image = image.transpose(2, 0, 1)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image))
        
        return dataset_dict