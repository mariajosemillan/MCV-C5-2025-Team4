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
"""
model.val(
            data="dataset.yaml",  # Ruta al archivo YAML del dataset
            split="val",  # División del dataset ('train', 'val' o 'test')
            batch=16,  # Tamaño del batch
            imgsz=640,  # Tamaño de las imágenes
            conf=0.001,  # Umbral de confianza mínimo para detección
            iou=0.6,  # Umbral de IoU para supresión de no máximos (NMS)
            device="cuda",  # Especificar dispositivo ('cuda' para GPU, 'cpu' para CPU)
            workers=8,  # Número de procesos de workers para la carga de datos
            save_json=True,  # Guarda los resultados en formato COCO JSON
            save_hybrid=False,  # Guarda imágenes con predicciones y etiquetas combinadas
            save_txt=False,  # Guarda predicciones en formato YOLO (txt)
            save_conf=False,  # Guarda las puntuaciones de confianza en los archivos de texto
            save_crop=False,  # Guarda los objetos detectados como imágenes recortadas
            save=True,  # Guarda imágenes con anotaciones
            half=False,  # Usa FP16 (media precisión) para mejorar rendimiento en GPU compatibles
            augment=False,  # Usa aumentación durante la validación
            rect=False,  # Utiliza redimensionado proporcional en validación
            vid_stride=1,  # Stride para videos (útil si validas sobre secuencias de video)
            plots=True,  # Genera y guarda gráficos de precisión, recall y mAP
            name="exp",  # Nombre del experimento en la carpeta de resultados
            exist_ok=False,  # Sobrescribe los resultados previos
            verbose=True,  # Muestra detalles adicionales en la salida
            project="runs/val",  # Carpeta donde se guardan los resultados
            classes=[0, 2],  # Filtra detecciones solo para ciertas clases (0=persona, 2=coche en COCO)
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
            classes=[0, 2],  # 0: person, 2: car
            device="cuda"  # Use GPU if available
        )

        # Move files after processing this batch
        move_files_from_batch_to_main(subdir)

def yolo_run_inference():
    """Run inference with YOLO."""
    try:
        # Load YOLOv8 model
        model = YOLO(MODEL_PATH)

        # Folders to process (train and val)
        # subdirs = ["train", "val"]
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

def yolo_run_eval():
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

    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)

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
    parser.add_argument("--mode", choices=["inference", "eval"], required=True, help="Choose whether to run inference or evaluation: 'inference' or 'eval'.")
    parser.add_argument("--model_name", type=str, default="yolov8n", help="Specify the YOLO model name (default: 'yolov8n').")
    args = parser.parse_args()

    # Assign model name and paths dynamically
    MODEL_NAME = args.model_name # YOLO11: yolo11x ; YOLO8: yolov8n
    MODEL_PATH = f"{MODEL_NAME}.pt"
    
    DATASET_DIR = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo"  # Input dataset directory
    OUTPUT_DIR = f"/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/output_{MODEL_NAME}"   # Directory where results will be saved

    # Configuration inference
    BATCH_SIZE = 32  # Number of images to process per batch
    # Configuration eval
    DATASET_CONFIG = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/dataset_yolo.yaml"

    # Run the selected mode
    if args.mode == "inference":
        yolo_run_inference()
    elif args.mode == "eval":
        yolo_run_eval()
