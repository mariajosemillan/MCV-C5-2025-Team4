"""
results = model.predict(
                        source="input.jpg",  # Imagen, video o carpeta con imágenes
                        conf=0.5,  # Umbral de confianza (por defecto 0.25)
                        iou=0.5,  # Umbral de IoU para supresión de no máximos (NMS)
                        save=True,  # Guardar resultados
                        save_txt=True,  # Guardar resultados en formato TXT
                        save_crop=True,  # Guardar los objetos recortados
                        show=True,  # Mostrar la imagen con predicciones
                        line_width=2,  # Grosor de las líneas de los bounding boxes
                        box=True,  # Dibujar cajas alrededor de los objetos
                        conf_label=True,  # Mostrar el valor de confianza en la predicción
                        device="cuda",  # Especificar GPU o CPU ("cpu" o "cuda")
                        classes=[0, 2],  # Filtrar por clases específicas (ej. 0: persona, 2: coche)
                        imgsz=640,  # Tamaño de la imagen de entrada (por defecto 640)
                        augment=True,  # Usar aumento de datos en inferencia
                        half=True  # Usar precisión de 16 bits (float16) para mayor velocidad en GPU
                    )
"""
import os
import sys
import shutil
from ultralytics import YOLO

import os
import sys
import shutil
from ultralytics import YOLO

# Configuration
MODEL_NAME = "yolo11x" # YOLO11: yolo11x ; YOLO8: yolov8n
MODEL_PATH = f"{MODEL_NAME}.pt"
DATASET_DIR = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo"  # Input dataset directory
OUTPUT_DIR = f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/output_{MODEL_NAME}"  # Directory where results will be saved
BATCH_SIZE = 32  # Number of images to process per batch

def move_files_from_batch_to_main(subdir):
    """Move all files from the current batch folder to the main folder."""
    main_folder = os.path.join(OUTPUT_DIR, "images")
    
    # Check if there are any new subfolders (train2, train3, etc.)
    subfolders = [folder for folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder))]
    print(f"SUBFOLDERS: {subfolders}")
    for subfolder in subfolders:
        if subfolder != 'train' and subfolder != 'val':
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
            classes=[0, 2],  # 0: person, 2: car
            device="cuda"  # Use GPU if available
        )

        # Move files after processing this batch
        move_files_from_batch_to_main(subdir)

def run_yolov8():
    """Run inference with YOLOv8."""
    try:
        # Load YOLOv8 model
        model = YOLO(MODEL_PATH)

        # Folders to process (train and val)
        subdirs = ["train", "val"]

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

if __name__ == "__main__":
    # Check if ultralytics is installed
    try:
        import ultralytics
    except ImportError:
        print("Ultralytics is not installed. Run:")
        print("pip install ultralytics")
        sys.exit(1)

    # Run inference with YOLOv8
    run_yolov8()
