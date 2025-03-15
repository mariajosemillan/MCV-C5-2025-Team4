
import os
from pycocotools import _mask  
import shutil
import glob 
import pycocotools.mask as maskUtils
import numpy as np
import cv2

def encode(bimask):
    '''Encodes a binary mask into RLE (Run-Length Encoding) format.

    This function checks the shape of the binary mask and then encodes it using 
    the pycocotools' mask encoding method. It handles both 2D and 3D binary masks.

    Args:
        bimask (numpy.ndarray): A binary mask, either 2D or 3D.

    Returns:
        dict: The Run-Length Encoding (RLE) representation of the input binary mask.
    '''
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]


def decode(rleObjs):
    '''Decodes RLE (Run-Length Encoding) objects into binary masks.

    This function decodes one or more RLE objects into binary masks, using the 
    pycocotools' mask decoding method. It ensures that the mask is in a 3D format 
    by selecting the first channel when necessary.

    Args:
        rleObjs (list or dict): A list or a single RLE object.

    Returns:
        numpy.ndarray: The decoded binary mask (or masks) in a 3D array.
    '''
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:, :, 0]

def toBbox(rleObjs):
    '''Converts an RLE (Run-Length Encoding) object to a bounding box.

    This function converts one or more RLE objects into bounding boxes, using 
    the pycocotools' mask toBbox method. The bounding box format is (xmin, ymin, 
    width, height).

    Args:
        rleObjs (list or dict): A list or a single RLE object.

    Returns:
        tuple: The bounding box coordinates (xmin, ymin, width, height) for the RLE object.
    '''
    if type(rleObjs) == list:
        return _mask.toBbox(rleObjs)
    else:
        return _mask.toBbox([rleObjs])[0]
    
def rle_to_polygons(rle):
    # Decodificar el RLE en una máscara binaria
    mask = maskUtils.decode(rle)
    # Asegurar que la máscara sea uint8 para `findContours`
    mask = (mask * 255).astype(np.uint8)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    # Convertir los contornos a formato COCO (lista de puntos x, y)
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()  # Convertir a lista de coordenadas x, y
        if len(contour) >= 6:  # Asegurar que sea un polígono válido (mínimo 3 puntos)
            polygons.append(contour)

    return polygons

def convert_kitti_mots_to_yolo(instance_txt_folder, output_folder, img_width=1242, img_height=375):
    '''Converts KITTI MOTS instance segmentation annotations to YOLO format.

    Args:
        instance_txt_folder (str): Path to the folder containing KITTI MOTS instance segmentation .txt files.
        output_folder (str): Path to the folder where the converted YOLO annotations will be saved.
        img_width (int, optional): Width of the images in the dataset. Defaults to 1224.
        img_height (int, optional): Height of the images in the dataset. Defaults to 370.

    The function reads instance segmentation annotations in the KITTI MOTS format, extracts relevant data, 
    and converts the encoded RLE (Run-Length Encoding) masks into YOLO bounding box format. The converted 
    annotations are stored in 'train' and 'val' subdirectories inside the output folder.

    Notes:
        - Sequences with IDs between 0 and 20 are considered for training and validation.
        - A predefined set of sequence IDs is allocated for validation.
        - The function normalizes bounding box coordinates for YOLO format.
        - 'DontCare' class IDs are filtered out.
    
    Outputs:
        - YOLO annotation files are saved in:
          - `{output_folder}/labels/train/` (for training sequences)
          - `{output_folder}/labels/val/` (for validation sequences)
    '''
    # List of sequence_ids for val (completed according to "MOTS: Multi-Object Tracking and Segmentation"[https://arxiv.org/pdf/1902.03604])
    val_sequence_ids = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]

    # Create output sub-folders
    train_folder = os.path.join(output_folder, "labels/train")
    val_folder = os.path.join(output_folder, "labels/val")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    for txt_file in os.listdir(instance_txt_folder):
        if not txt_file.endswith('.txt'):
            continue
        
        sequence_id = os.path.splitext(txt_file)[0]  # Extract sequence_id from filename

        # Define in which folder to save -- sequences from 0 to 20 are for train + val (12 for train + 9 for val)
        if int(sequence_id) <= 20:
            if sequence_id in val_sequence_ids:
                target_folder = val_folder
            else:
                target_folder = train_folder
        else:
            print(f"WARNING: {sequence_id} is not in train or val, will be omitted.")
            continue

        txt_path = os.path.join(instance_txt_folder, txt_file)

        # Dictionary to hold lines per img_id
        img_data = {}
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()

                img_id = int(parts[0])  # First column is img_id
                track_id = int(parts[1]) # Second column is track id (not used here)
                class_id = int(parts[2])  # 1 = Car, 2 = Pedestrian
                if class_id == 1:
                    class_id = 2 # 2 = Car in Yolo
                elif class_id == 2:
                    class_id = 0 # 0 = Person in Yolo
                rle_data = {
                    'size': [int(parts[3]), int(parts[4])],
                    'counts': parts[5].encode('utf-8')
                }

                if class_id != 10: # filter dont care ids
                    img_width=int(parts[4])
                    img_height=int(parts[3])
                    polygons = rle_to_polygons(rle_data)
                    for polygon in polygons:
                        # Normalizar los puntos del polígono
                        normalized_polygon = [
                            (x / img_width, y / img_height) for x, y in zip(polygon[::2], polygon[1::2])
                        ]
                        
                        # Crear la línea de segmentación en el formato YOLO (que usa los puntos del polígono)
                        yolo_line = f'{class_id} ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in normalized_polygon) + '\n'
                        if img_id not in img_data:
                            img_data[img_id] = []
                        img_data[img_id].append(yolo_line)


        # Save each img_id lines on its corresponding file
        for img_id, lines in img_data.items():
            new_filename = f"{sequence_id}_{img_id:06d}.txt"
            new_path = os.path.join(target_folder, new_filename)

            with open(new_path, 'w') as out_f:
                out_f.writelines(lines)

    print("Conversion completed.")


def create_folders(base_path):
    '''Creates the necessary folder structure for the YOLO dataset.

    Args:
        base_path (str): Path where the folder structure for images and labels will be created.
    
    The function creates the following directory structure under `base_path`:
        - `images/train`, `images/val`, `images/test` for storing images in respective splits.
        - `labels/train`, `labels/val` for storing labels for the training and validation splits.
    
    Folders will be created if they don't already exist, ensuring the directory structure is set up for YOLO annotations.

    Notes:
        - This function does not create directories for `test/labels` since labels for testing are not required in YOLO format.
    '''
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_path, 'images', split), exist_ok=True)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_path, 'labels', split), exist_ok=True)


def process_images_and_labels(base_dir, output_dir_name, instance_txt_dir, img_width=1242, img_height=375):
    '''Processes images and labels from the KITTI MOTS dataset and converts them to YOLO format.

    Args:
        base_dir (str): Path to the root directory containing the KITTI MOTS dataset.
        output_dir_name (str): Name of the directory where the processed dataset will be stored.
        instance_txt_dir (str): Path to the folder containing instance segmentation annotation files.
        img_width (int, optional): Width of the images in the dataset. Defaults to 1224.
        img_height (int, optional): Height of the images in the dataset. Defaults to 370.

    The function performs the following steps:
        1. Creates the necessary folder structure for the YOLO dataset.
        2. Copies images from the KITTI MOTS dataset into the appropriate YOLO directories:
            - `train` images from "training"
            - `val` images from "validating"
            - `test` images from "testing"
        3. Renames images to include their sequence ID.
        4. Converts KITTI MOTS segmentation labels to YOLO format using `convert_kitti_mots_to_yolo`.
        5. Saves the processed labels and images into the designated output directory.

    Notes:
        - If a split directory (`train`, `val`, or `test`) does not exist, it will be skipped.
        - The function ensures that YOLO-formatted labels are saved in the correct structure.
        - Output images and labels are stored under `{base_dir}/{output_dir_name}/images/` and `{base_dir}/{output_dir_name}/labels/`.

    Outputs:
        - YOLO-formatted images and labels in `{base_dir}/{output_dir_name}/`.
    '''
    dataset_path = os.path.join(base_dir, output_dir_name)
    create_folders(dataset_path)
    
    # Create YOLO directories structure and copy images
    for split in ['train', 'val', 'test']:
        # Original data
        split_path = os.path.join(base_dir, 
                                  "training" if split == "train" else 
                                  "validating" if split == "val" else 
                                  "testing")
        # New directory for images
        dest_img_path = os.path.join(dataset_path, 'images', split)

        if not os.path.exists(split_path):
            print(f"Directory {split_path} does not exist, skipping...")
            continue

        print(f"Processing images from {split_path} to {dest_img_path}")

        # Run through sequence subdirs
        for sequence_id in sorted(os.listdir(split_path)):
            subdir_path = os.path.join(split_path, sequence_id)
            if os.path.isdir(subdir_path):  # Verify if its a dir
                images = glob.glob(os.path.join(subdir_path, "*.png")) # Search for images
                for image in images:
                    image_id = os.path.basename(image) # Unique name based on the origin directory -- image_id
                    new_filename = f"{sequence_id}_{image_id}"
                    destino = os.path.join(dest_img_path, new_filename)
                    shutil.copy(image, destino) # Copy file
    
    # Process labels -- convert from kitti to yolo and store with the right directory structure
    convert_kitti_mots_to_yolo(instance_txt_dir, dataset_path, img_width, img_height)
    
    print(f"Images and labels converted from Kitti to Yolo format.\nOutput files can be found at: {dataset_path}")


if __name__ == "__main__":
    # Path to dir containing Kitti trainig, val and testing datasets
    base_dir = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset"
    # Path to Kitti txt files dir 'instance_txt'
    instance_txt_dir = '/ghome/c5mcv04/mcv/datasets/C5/KITTI-MOTS/instances_txt'
    # Dir name for YOLO dataset
    output_dir_name = 'dataset_yolo_masks_sytfvivii'
    # Convert dataset from Kitti format to YOLO format and directory structure
    process_images_and_labels(base_dir, output_dir_name, instance_txt_dir)
